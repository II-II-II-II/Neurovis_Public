import pandas as pd
import io
import json
import warnings
import time
import asyncio
from datetime import datetime
from typing import List

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from starlette.datastructures import UploadFile as StarletteUploadFile
import uvicorn
import sys

# --- THE MODULAR IMPORTS ---
import HRVAgent
import neuroAgent

from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import ChatMessage

warnings.filterwarnings('ignore')

app = FastAPI()

# ==========================================
# ROUTER STATE & LOGGER
# ==========================================
chat_history = []

class LoggerWriter:
    def __init__(self, filename):
        self.file = open(filename, 'a', encoding='utf-8')
    def write(self, message):
        self.file.write(message)
        self.file.flush()
    def flush(self):
        pass

def log_action(agent_name: str, action: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{agent_name}] {action}\n"
    print(log_entry.strip())
    with open("neurovis_agent_thoughts.log", "a", encoding="utf-8") as f:
        f.write(log_entry)

# ==========================================
# AI ENGINE CONFIGURATION
# ==========================================
print("Initializing Agent Orchestrator...")
llm = Ollama(model="qwen2.5", request_timeout=360.0) 
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

@app.on_event("startup")
async def startup_event():
    log_action("SYSTEM", "Triggering HRVAgent internal startup sequence...")
    if hasattr(HRVAgent, 'startup_event'):
        await HRVAgent.startup_event()

# ==========================================
# ROUTER TOOLS (The Handoffs)
# ==========================================
async def call_macro_biometrician(query: str) -> str:
    log_action("ROUTER", f"Routing query to Macro Biometrician (HRVAgent): {query}")
    try:
        response = await HRVAgent.agent.run(user_msg=query, chat_history=[])
        if isinstance(response, dict) and "response" in response:
            final_answer = str(response["response"])
        else:
            final_answer = str(response)
        log_action("HRV_AGENT", f"Final response generated: {final_answer}")
        return final_answer
    except Exception as e:
        log_action("HRV_AGENT", f"Failed: {str(e)}")
        return f"HRV Agent encountered an error: {str(e)}"

def call_macro_biometrician_sync(query: str) -> str:
    return asyncio.run(call_macro_biometrician(query))

async def call_neuro_analyst(query: str) -> str:
    log_action("ROUTER", f"Routing query to Neuro Analyst: {query}")
    try:
        response = await neuroAgent.neuro_agent.run(user_msg=query, chat_history=[])
        if isinstance(response, dict) and "response" in response:
            final_answer = str(response["response"])
        else:
            final_answer = str(response)
        log_action("NEURO_AGENT", f"Final response generated: {final_answer}")
        return final_answer
    except Exception as e:
        log_action("NEURO_AGENT", f"Failed: {str(e)}")
        return f"Neuro Agent encountered an error: {str(e)}"

def call_neuro_analyst_sync(query: str) -> str:
    return asyncio.run(call_neuro_analyst(query))

router_tools = [
    FunctionTool.from_defaults(
        fn=call_macro_biometrician_sync,
        async_fn=call_macro_biometrician,
        name="call_macro_biometrician",
        description="Use this tool when the user asks about Apple Watch data, sleep duration, daily HRV trends, workouts, or general long-term baselines."
    ),
    FunctionTool.from_defaults(
        fn=call_neuro_analyst_sync,
        async_fn=call_neuro_analyst,
        name="call_neuro_analyst",
        description="Use this tool when the user asks about specific meditation sessions, Engagement/Flow/Vigilance/Detachment states, Frontal Alpha Asymmetry (FAA), or EEG brainwaves."
    )
]

ROUTER_PROMPT = f"""
You are the Neurovis Triage Router. Your job is to analyze the user's question and route it to the correct specialized agent.
Today's date is {datetime.now().strftime("%Y-%m-%d")}.

DATA ROUTING RULES:
1. If the user asks about Apple Watch data, sleep duration, daily HRV trends, or workouts, use the `call_macro_biometrician` tool.
2. If the user asks about specific meditation sessions, Engagement/Flow/Vigilance/Detachment states, Frontal Alpha Asymmetry, or EEG brainwaves, use the `call_neuro_analyst` tool.

AMBIGUITY RULE:
If the user asks for a metric like "HRV" or "Heart Rate" that exists in BOTH Apple Watch data AND Meditation data, you MUST ask the user to clarify which source they want to analyze BEFORE calling any tools. Do not guess.

CRITICAL DATE FORMATTING:
If the user asks about a timeframe (e.g., "last 7 days", "March", "yesterday"), you MUST translate it into a strict YYYY-MM-DD format using today's date ({datetime.now().strftime("%Y-%m-%d")}) before passing the query.

ERROR HANDLING:
- If a tool returns an error saying a column is missing or data is not found, DO NOT invent an excuse. 
- Instead, tell the user: "The required data file for this query was not uploaded. Please ensure you have uploaded the correct Apple Health or Neurovis session file."

Select the appropriate tool and pass the translated question. Once the tool returns a response, output that exact response to the user.
"""

router_agent = ReActAgent(
    tools=router_tools, 
    llm=llm, 
    system_prompt=ROUTER_PROMPT,
    verbose=True,
    max_iterations=10
)

# ==========================================
# FASTAPI ENDPOINTS (API GATEWAY FOR FILE UPLOADS)
# ==========================================
@app.post("/upload_context")
async def upload_context(files: List[UploadFile] = File(...)):
    loaded_files = []
    
    for file in files:
        contents = await file.read()
        filename = file.filename.lower()
        
        # 1. The Raw Details JSON goes directly to HRVAgent's native upload endpoint
        if filename.endswith('.json') and 'raw_details' in filename:
            hrv_file = StarletteUploadFile(filename=file.filename, file=io.BytesIO(contents))
            response = await HRVAgent.upload_data(file=hrv_file)
            log_action("SYSTEM", f"Routed JSON payload natively to HRVAgent: {filename} | Status: {response}")
            loaded_files.append(filename)
            
        # 2. The Session Backup JSON goes to neuroAgent
        elif filename.endswith('.json') and 'raw_details' not in filename:
            neuro_file = StarletteUploadFile(filename=file.filename, file=io.BytesIO(contents))
            await neuroAgent.upload_context(files=[neuro_file])
            log_action("SYSTEM", f"Routed Session JSON natively to neuroAgent: {filename}")
            loaded_files.append(filename)
            
        # 3. The Neuro CSV goes to neuroAgent
        elif filename.endswith('.csv') and 'apple' not in filename:
            neuro_file = StarletteUploadFile(filename=file.filename, file=io.BytesIO(contents))
            await neuroAgent.upload_context(files=[neuro_file])
            log_action("SYSTEM", f"Routed Neuro CSV natively to neuroAgent: {filename}")
            loaded_files.append(filename)
            
        # 4. Fallback for Apple Health CSV directly into HRVAgent
        elif filename.endswith('.csv') and 'apple' in filename:
            df = pd.read_csv(io.BytesIO(contents))
            HRVAgent.df_hrv = df
            if 'timestamp_utc' in df.columns:
                HRVAgent.df_hrv['timestamp'] = pd.to_datetime(df['timestamp_utc'])
                HRVAgent.df_hrv['date'] = HRVAgent.df_hrv['timestamp'].dt.date
            HRVAgent.available_hrv_columns = list(HRVAgent.df_hrv.columns)
            log_action("SYSTEM", f"Loaded Apple Health CSV directly into HRVAgent: {filename}")
            loaded_files.append(filename)
            
    return {"message": f"Successfully loaded and routed natively: {', '.join(loaded_files)}"}

class ChatRequest(BaseModel):
    message: str

@app.get("/")
async def serve_ui():
    """Serves the main Neurovis HTML interface."""
    return FileResponse("neurovisAgent.html")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    global router_agent, chat_history
    log_action("ROUTER", f"Received user prompt: {request.message}")
    
    start_time = time.time()
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open("neurovis_agent_thoughts.log", "a", encoding="utf-8") as f:
        f.write(f"\n\n{'='*70}\n")
        f.write(f"[{timestamp_str}] NEW REQUEST\n")
        f.write(f"USER PROMPT: {request.message}\n")
        f.write(f"{'-'*70}\nAGENT THOUGHT PROCESS:\n")
    
    original_stdout = sys.stdout
    sys.stdout = LoggerWriter("neurovis_agent_thoughts.log")
    
    try:
        response = await router_agent.run(user_msg=request.message, chat_history=chat_history)
        
        if isinstance(response, dict) and "response" in response:
            final_answer = str(response["response"])
        else:
            final_answer = str(response)
            
        chat_history.append(ChatMessage(role="user", content=request.message))
        chat_history.append(ChatMessage(role="assistant", content=final_answer))
        
        elapsed_time = time.time() - start_time
        sys.stdout = original_stdout
        
        with open("neurovis_agent_thoughts.log", "a", encoding="utf-8") as f:
            f.write(f"\n{'-'*70}\n")
            f.write(f"FINAL RESPONSE (Generated in {elapsed_time:.2f} seconds):\n{final_answer}\n")
            f.write(f"{'='*70}\n")
        
        log_action("ROUTER", f"Request completed in {elapsed_time:.2f}s")
        return {"reply": final_answer}
        
    except Exception as e:
        sys.stdout = original_stdout
        elapsed_time = time.time() - start_time
        
        with open("neurovis_agent_thoughts.log", "a", encoding="utf-8") as f:
            f.write(f"\n{'-'*70}\n")
            f.write(f"CRITICAL ERROR (Failed after {elapsed_time:.2f} seconds):\n{str(e)}\n")
            f.write(f"{'='*70}\n")
            
        log_action("ROUTER", f"CRITICAL ERROR: {str(e)}")
        return {"reply": f"An internal error occurred: {str(e)}"}
    finally:
        sys.stdout = original_stdout

if __name__ == "__main__":
    log_action("SYSTEM", "Booting Payload-Routing API Gateway...")
    uvicorn.run(app, host="0.0.0.0", port=8000)