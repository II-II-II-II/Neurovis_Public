import pandas as pd
import numpy as np
import warnings
import io
import json
from datetime import datetime
from starlette.datastructures import UploadFile

from llama_index.llms.ollama import Ollama
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent

warnings.filterwarnings('ignore')

# ==========================================
# AGENT STATE
# ==========================================
df_neuro_csv = pd.DataFrame()    # High-frequency single session data
df_macro = pd.DataFrame()        # Longitudinal database from JSON
df_macro_chunks = pd.DataFrame() # Historical minute-by-minute chunk data

# ==========================================
# AI ENGINE CONFIGURATION
# ==========================================
print(" Initializing Advanced Neuro Analyst Engine...")
llm = Ollama(model="qwen2.5", request_timeout=360.0) 

def log_action(agent_name: str, action: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{agent_name}] {action}\n"
    print(log_entry.strip())
    with open("neurovis_agent_thoughts.log", "a", encoding="utf-8") as f:
        f.write(log_entry)

# ==========================================
# API RECEIVER & DATA PARSER
# ==========================================
async def upload_context(files: list[UploadFile]):
    """Receives raw file payloads and structures them into robust Pandas DataFrames."""
    global df_neuro_csv, df_macro, df_macro_chunks
    for file in files:
        contents = await file.read()
        filename = file.filename.lower()
        
        if filename.endswith('.csv'):
            df_neuro_csv = pd.read_csv(io.BytesIO(contents))
            # Standardize timestamp for CSV
            if 'timestamp' in df_neuro_csv.columns:
                df_neuro_csv['parsed_time'] = pd.to_datetime(df_neuro_csv['timestamp'], format='mixed', errors='coerce')
            log_action("NEURO_AGENT", f"Successfully ingested and structured CSV payload: {filename}")
            
        elif filename.endswith('.json'):
            parsed = json.loads(contents.decode('utf-8'))
            
            # 1. Parse high-level session summaries
            if 'sessions' in parsed:
                # Flatten the JSON into a robust DataFrame
                sessions_list = []
                for s in parsed['sessions']:
                    flat_s = {'timestamp': s.get('timestamp'), 'session_id': s.get('session_id')}
                    summary = s.get('summary', {})
                    flat_s.update(summary)
                    sessions_list.append(flat_s)
                
                df_macro = pd.DataFrame(sessions_list)
                
                # Dynamically Engineer FBA from L/R Beta Channels
                if 'avg_r_beta' in df_macro.columns and 'avg_l_beta' in df_macro.columns:
                    df_macro['avg_fba'] = df_macro['avg_r_beta'] - df_macro['avg_l_beta']
                
                # Engineer Time-Series Features
                if 'timestamp' in df_macro.columns:
                    df_macro['parsed_time'] = pd.to_datetime(df_macro['timestamp'], errors='coerce', utc=True).dt.tz_localize(None)
                    df_macro['date'] = df_macro['parsed_time'].dt.date
                    df_macro['day_of_week'] = df_macro['parsed_time'].dt.day_name()
                    df_macro['hour'] = df_macro['parsed_time'].dt.hour
                    df_macro['time_of_day'] = pd.cut(
                        df_macro['hour'], 
                        bins=[0, 6, 12, 18, 24], 
                        labels=['Night', 'Morning', 'Afternoon', 'Evening'], 
                        right=False
                    )
                    
            # 2. Parse minute-by-minute climate chunks
            if 'climate_chunks' in parsed:
                chunks_list = []
                for session_chunks in parsed['climate_chunks']:
                    sess_id = session_chunks.get('session_id')
                    for c in session_chunks.get('chunks', []):
                        c_flat = c.copy()
                        c_flat['session_id'] = sess_id
                        chunks_list.append(c_flat)
                df_macro_chunks = pd.DataFrame(chunks_list)
                
            log_action("NEURO_AGENT", f"Successfully converted JSON into Macro Time-Series DataFrames: {filename}")

# ==========================================
# TOOL 1: MICRO SESSION SUMMARY
# ==========================================
def micro_session_summary() -> str:
    """
    Provides a high-level overview of the currently loaded single session CSV.
    Use this to get the session date, average FAA, average HRV, and to check which devices were active.
    """
    global df_neuro_csv
    if df_neuro_csv.empty: return "No single session CSV loaded."
    
    results = ["--- Single Session Summary ---"]
    
    # Check for Date
    if 'parsed_time' in df_neuro_csv.columns and not df_neuro_csv['parsed_time'].dropna().empty:
        session_date = df_neuro_csv['parsed_time'].dropna().iloc[0].strftime('%Y-%m-%d')
        results.append(f"Session Date: {session_date}")
    else:
        results.append("Session Date: Unknown (No valid timestamp found in CSV).")
    
    # Check EEG Data
    if 'raw_faa' in df_neuro_csv.columns:
        avg_faa = df_neuro_csv['raw_faa'].mean()
        results.append(f"EEG Active: Yes. Average FAA: {avg_faa:.4f}")
    else:
        results.append("EEG Active: No (FAA data missing).")
        
    # Check HRV Data
    if 'rmssd' in df_neuro_csv.columns:
        avg_hrv = df_neuro_csv['rmssd'].mean()
        results.append(f"HRV Active: Yes. Average HRV (rMSSD): {avg_hrv:.2f} ms")
    else:
        results.append("HRV Active: No (rMSSD data missing).")
        
    return "\n".join(results)

# ==========================================
# TOOL 2: MICRO CORRELATION & TRENDS
# ==========================================
def micro_correlation_and_trends(metric: str) -> str:
    """
    Breaks down a session minute-by-minute, calculates correlation between FAA and HRV, 
    and determines if a metric was increasing or decreasing over time.
    Valid inputs for metric: 'faa', 'hrv', 'correlation'.
    """
    global df_neuro_csv
    if df_neuro_csv.empty or 'parsed_time' not in df_neuro_csv.columns:
        return "No session data or timestamps available."
        
    df = df_neuro_csv.dropna(subset=['parsed_time'])
    df['minute'] = df['parsed_time'].dt.floor('Min')
    
    if metric == 'correlation':
        if 'raw_faa' in df.columns and 'rmssd' in df.columns:
            valid_df = df.dropna(subset=['raw_faa', 'rmssd'])
            if len(valid_df) < 2: return "Not enough overlapping data points to calculate correlation."
            corr = valid_df['raw_faa'].corr(valid_df['rmssd'])
            return f"The Pearson correlation coefficient between FAA and HRV is {corr:.3f}. (1 is perfectly positive, -1 is perfectly negative)."
        else:
            return "Cannot calculate correlation. Missing either FAA or HRV data."
            
    elif metric in ['faa', 'hrv']:
        col_map = {'faa': 'raw_faa', 'hrv': 'rmssd'}
        target_col = col_map[metric]
        
        if target_col not in df.columns:
            return f"Data for {metric} not found in this session."
            
        min_avg = df.groupby('minute')[target_col].mean().dropna()
        if min_avg.empty: return "No valid data to group."
        
        x = np.arange(len(min_avg))
        y = min_avg.values
        slope, _ = np.polyfit(x, y, 1)
        trend = "INCREASING" if slope > 0 else "DECREASING"
        
        results = [f"--- Minute-by-Minute Breakdown for {metric.upper()} ---"]
        results.append(f"Overall Trend: {trend} (Slope: {slope:.4f})")
        
        for time, val in min_avg.items():
            results.append(f"- {time.strftime('%H:%M')}: {val:.4f}")
            
        return "\n".join(results)
        
    return "Invalid metric requested."

# ==========================================
# TOOL 3: MICRO ADVANCED EEG
# ==========================================
def micro_advanced_eeg() -> str:
    """
    Calculates experimental and advanced biometrics from a single session like Functional Beta Asymmetry.
    """
    global df_neuro_csv
    if df_neuro_csv.empty: return "No single session CSV loaded."
    
    required_cols = ['L_beta', 'R_beta', 'L_theta', 'R_theta']
    if not all(col in df_neuro_csv.columns for col in required_cols):
        return "Advanced EEG requires L/R Beta and Theta channels. Data is missing."
        
    df = df_neuro_csv.dropna(subset=required_cols)
    
    # Functional Beta Asymmetry (R_beta - L_beta) -> Higher means more right beta (anxiety/active)
    fba = (df['R_beta'] - df['L_beta']).mean()
    
    # Theta/Beta Ratio (TBR) -> Often used for focus/relaxation metrics
    avg_theta = (df['L_theta'] + df['R_theta']) / 2
    avg_beta = (df['L_beta'] + df['R_beta']) / 2
    tbr = (avg_theta / avg_beta).mean()
    
    return f"Advanced Metrics:\n- Functional Beta Asymmetry (FBA): {fba:.4f} (Positive = Right Dominant, Negative = Left Dominant)\n- Average Theta/Beta Ratio (TBR): {tbr:.4f}"

# ==========================================
# TOOL 3.5: MACRO CHRONOLOGICAL LISTER
# ==========================================
def macro_chronological_lister(metric: str) -> str:
    """
    Analyzes historical meditation sessions and lists the specific metric broken down chronologically by exact date.
    Valid metrics: 'avg_faa', 'approach_pct', 'avg_hrv', 'avg_hr', 'avg_l_beta', 'avg_r_beta', 'avg_fba'.
    Note: Theta/TBR is NOT recorded historically.
    """
    global df_macro
    if df_macro.empty: 
        return "No historical database loaded."
    
    if metric not in df_macro.columns:
        return f"No data found for metric '{metric}'. Note: Historical Theta/TBR is unavailable."
        
    # Drop rows that are missing the metric, and sort chronologically by time
    df_valid = df_macro.dropna(subset=['parsed_time', metric]).sort_values('parsed_time')
    
    if df_valid.empty:
        return f"No historical records found for {metric}."
        
    results = [f"Chronological breakdown for {metric}:"]
    
    # Iterate through the DataFrame and format the output
    for _, row in df_valid.iterrows():
        # Using YYYY-MM-DD format for clean output
        date_str = row['parsed_time'].strftime('%Y-%m-%d')
        val = row[metric]
        
        # Format floats to 4 decimal places
        if isinstance(val, (int, float)):
            results.append(f"- {date_str}: {val:.4f}")
        else:
            results.append(f"- {date_str}: {val}")
            
    return "\n".join(results)

# ==========================================
# TOOL 4: MACRO TREND ANALYZER
# ==========================================
def macro_trend_analyzer(metric: str, analysis_type: str) -> str:
    """
    Analyzes historical trends from the JSON database across multiple sessions.
    Valid metrics: 'avg_faa', 'avg_hrv', 'avg_hr', 'avg_l_beta', 'avg_r_beta', 'avg_fba'.
    Note: Theta/TBR is NOT recorded historically.
    Valid analysis_type: 'time_of_day' (Morning vs Evening), 'day_of_week' (Mon vs Tue), 'overall_trend' (Degradation vs Improvement over time).
    """
    global df_macro
    if df_macro.empty: return "No historical macro database loaded."
    if metric not in df_macro.columns: return f"Metric '{metric}' not found in historical data."
    
    df = df_macro.dropna(subset=[metric, 'parsed_time'])
    if df.empty: return f"No valid historical records found for {metric}."
    
    if analysis_type == 'time_of_day':
        grouped = df.groupby('time_of_day')[metric].mean().dropna()
        return f"Historical Averages by Time of Day for {metric}:\n" + "\n".join([f"- {k}: {v:.4f}" for k, v in grouped.items()])
        
    elif analysis_type == 'day_of_week':
        grouped = df.groupby('day_of_week')[metric].mean().dropna()
        return f"Historical Averages by Day of Week for {metric}:\n" + "\n".join([f"- {k}: {v:.4f}" for k, v in grouped.items()])
        
    elif analysis_type == 'overall_trend':
        df_sorted = df.sort_values('parsed_time')
        x = np.arange(len(df_sorted))
        y = df_sorted[metric].values
        
        if len(y) < 2: return "Not enough historical data to calculate a trend."
        
        slope, _ = np.polyfit(x, y, 1)
        direction = "IMPROVING/INCREASING" if slope > 0 else "DEGRADING/DECREASING"
        
        return f"Longitudinal Trend Analysis for {metric}:\n- Over {len(y)} sessions, the metric is {direction} (Slope: {slope:.4f}).\n- First recorded value: {y[0]:.4f}\n- Most recent value: {y[-1]:.4f}"

    return "Invalid analysis type requested."

# ==========================================
# TOOL 5: MACRO HISTORICAL CHUNK TRENDS
# ==========================================
def macro_historical_chunk_trends(metric: str) -> str:
    """
    Analyzes the historical minute-by-minute data across all past sessions to find the average curve of a metric over time.
    Use this to compare a session's minute-by-minute breakdown against historical minute-by-minute averages.
    Valid metrics: 'avg_hr', 'avg_hrv', 'avg_faa', 'avg_l_beta', 'avg_r_beta'.
    """
    global df_macro_chunks
    if df_macro_chunks.empty:
        return "No historical minute-by-minute chunk data loaded."
    if metric not in df_macro_chunks.columns:
        return f"Metric '{metric}' not found in historical chunk data."
        
    grouped = df_macro_chunks.groupby('chunk_index')[metric].mean().dropna()
    if grouped.empty:
        return "No valid chunk data to group."
        
    results = [f"Average historical session curve for {metric} by minute (chunk):"]
    for minute, val in grouped.head(30).items():
        results.append(f"- Minute {minute}: {val:.4f}")
        
    return "\n".join(results)

# ==========================================
# AGENT REGISTRATION
# ==========================================
neuro_tools = [
    FunctionTool.from_defaults(fn=micro_session_summary),
    FunctionTool.from_defaults(fn=micro_correlation_and_trends),
    FunctionTool.from_defaults(fn=micro_advanced_eeg),
    FunctionTool.from_defaults(fn=macro_chronological_lister),
    FunctionTool.from_defaults(fn=macro_trend_analyzer),
    FunctionTool.from_defaults(fn=macro_historical_chunk_trends)
]

SYSTEM_PROMPT = """
You are the Advanced Neuro Analyst. You analyze meditation data, specializing in both granular single-session CSV data and historical JSON trends.

CRITICAL INSTRUCTIONS:
1. If the user asks about "this session", "this morning", or minute-by-minute details, use the `micro_` tools.
2. If the user asks about history, trends over time, days of the week, or overall improvement/degradation, use the `macro_` tools.
3. If data is missing (e.g., HRV is missing because they only wore the Muse), explain this clearly based on the tool's output without guessing.
"""

neuro_agent = ReActAgent(
    tools=neuro_tools, 
    llm=llm, 
    system_prompt=SYSTEM_PROMPT,
    verbose=True,
    max_iterations=10
)