import pandas as pd
import glob
import os
import asyncio
import json
import scipy.stats as stats
import warnings

from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.agent import ReActAgent
from typing import Literal
from llama_index.core.llms import ChatMessage
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# Suppress pandas chained assignment warnings for cleaner terminal output
warnings.filterwarnings('ignore')

# Global DataFrames
df_hrv = pd.DataFrame()
df_workouts = pd.DataFrame()
available_hrv_columns = []
available_workout_columns = []

# --- PART 1: THE ROBUST PARAMETERIZED TOOLS ---

def generate_biometric_chart(metric1: str, metric2: str = None) -> str:
    """Generates a line graph of the requested metrics over time and saves it as an image."""
    global df_hrv, df_workouts
    try:
        if metric1 in available_hrv_columns:
            working_df = df_hrv.dropna(subset=['timestamp', metric1]).sort_values('timestamp')
        elif metric1 in available_workout_columns:
            working_df = df_workouts.dropna(subset=['timestamp', metric1]).sort_values('timestamp')
        else:
            return f"Error: Metric {metric1} not found in data."

        plt.figure(figsize=(12, 6))
        plt.plot(working_df['timestamp'], working_df[metric1], label=metric1, alpha=0.7, color='#007aff')
        
        if metric2:
            if metric2 in available_hrv_columns:
                working_df2 = df_hrv.dropna(subset=['timestamp', metric2]).sort_values('timestamp')
                plt.plot(working_df2['timestamp'], working_df2[metric2], label=metric2, alpha=0.7, color='#34c759')
            elif metric2 in available_workout_columns:
                working_df2 = df_workouts.dropna(subset=['timestamp', metric2]).sort_values('timestamp')
                plt.plot(working_df2['timestamp'], working_df2[metric2], label=metric2, alpha=0.7, color='#34c759')

        plt.title(f"{metric1} {f'and {metric2}' if metric2 else ''} Over Time")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filename = "neurovis_chart_output.png"
        plt.savefig(filename)
        plt.close()
        
        return f"Successfully generated a chart and saved it locally as '{filename}'. Tell the user to open this file."
        
    except Exception as e:
        return f"Failed to generate chart: {e}"

def analyze_biometric_data(
    metric: str, 
    analysis_type: Literal[
        'count', # <--- 1. ADD 'count' HERE
        'highest_date', 'lowest_date', 
        'overall_max', 'overall_min', 'overall_average', 'correlation', 
        'trend_slope', 'day_of_week_average'
    ], 
    metric2: str = None,
    start_date: str = None,
    end_date: str = None
) -> str:
    global df_hrv, df_workouts
    try:
        if metric in available_hrv_columns:
            working_df = df_hrv.copy()
        elif metric in available_workout_columns:
            working_df = df_workouts.copy()
        else:
            return f"Error: '{metric}' is not a valid column."
        if start_date and end_date:
            try:
                # Convert both dates to standard YYYY-MM-DD
                s_date = pd.to_datetime(start_date).date()
                e_date = pd.to_datetime(end_date).date()
                working_df = working_df[(working_df['date'] >= s_date) & (working_df['date'] <= e_date)]
                if working_df.empty:
                    return f"Error: No data found between {start_date} and {end_date}."
            except Exception as e:
                return f"Error parsing dates. Ensure format is YYYY-MM-DD. Details: {e}"

        if analysis_type == 'count':
            return f"There are {len(working_df)} total data points for {metric} in this period."

        if analysis_type == 'correlation':
            if not metric2:
                return "Error: You must provide 'metric2'."
            if metric2 not in working_df.columns:
                return f"Error: '{metric2}' must be in the same dataset."
            corr_val = working_df[metric].corr(working_df[metric2])
            return f"The statistical correlation between {metric} and {metric2} is {corr_val:.3f}."

        elif analysis_type == 'highest_date':
            max_idx = working_df[metric].idxmax()
            return f"The highest {metric} was {working_df.loc[max_idx, metric]:.2f} on {working_df.loc[max_idx, 'timestamp']}"

        elif analysis_type == 'lowest_date':
            min_idx = working_df[metric].idxmin()
            return f"The lowest {metric} was {working_df.loc[min_idx, metric]:.2f} on {working_df.loc[min_idx, 'timestamp']}"
            
        elif analysis_type == 'day_of_week_average':
            working_df['day_of_week'] = working_df['timestamp'].dt.day_name()
            day_avgs = working_df.groupby('day_of_week')[metric].mean().round(2).to_dict()
            return f"Average {metric} by day of week: {day_avgs}"
            
        elif analysis_type == 'trend_slope':
            midpoint = len(working_df) // 2
            first_half = working_df[metric].iloc[:midpoint].mean()
            second_half = working_df[metric].iloc[midpoint:].mean()
            diff = second_half - first_half
            direction = "increasing" if diff > 0 else "decreasing"
            return f"Statistically, {metric} is {direction} over time by an average shift of {abs(diff):.2f}."

        elif analysis_type == 'overall_max':
            return f"The absolute maximum recorded {metric} is {working_df[metric].max():.2f}"
            
        elif analysis_type == 'overall_min':
            return f"The absolute minimum recorded {metric} is {working_df[metric].min():.2f}"
            
        elif analysis_type == 'overall_average':
            return f"The overall average {metric} is {working_df[metric].mean():.2f}"
            
        else:
            return f"Error: {analysis_type} is not supported."

    except Exception as e:
        return f"A data calculation error occurred: {str(e)}"

def calculate_statistical_significance(
    metric: str, 
    start_date_1: str, end_date_1: str, 
    start_date_2: str, end_date_2: str
) -> str:
    """Calculates the p-value (statistical significance) of a metric between two date ranges."""
    global df_hrv, df_workouts
    try:
        if metric in available_hrv_columns:
            working_df = df_hrv.copy()
        elif metric in available_workout_columns:
            working_df = df_workouts.copy()
        else:
            return f"Error: '{metric}' is not a valid column."

        s1, e1 = pd.to_datetime(start_date_1).date(), pd.to_datetime(end_date_1).date()
        s2, e2 = pd.to_datetime(start_date_2).date(), pd.to_datetime(end_date_2).date()

        group1 = working_df[(working_df['date'] >= s1) & (working_df['date'] <= e1)][metric].dropna()
        group2 = working_df[(working_df['date'] >= s2) & (working_df['date'] <= e2)][metric].dropna()

        if len(group1) < 2 or len(group2) < 2:
            return "Not enough data points to calculate statistical significance."

        t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
        
        significance = "Statistically Significant" if p_val < 0.05 else "Not Statistically Significant"
        return f"The p-value for {metric} between the two periods is {p_val:.4f}. This is {significance}."
    except Exception as e:
        return f"Failed to calculate p-value: {e}"

# --- DEEP ANALYSIS PANDAS ENGINE ---
def get_deep_period_stats(start_date_str, end_date_str):
    """Generates a massive statistical payload and returns raw data for differential math."""
    global df_hrv, df_workouts
    
    start = pd.to_datetime(start_date_str, format="%m-%d-%Y").date()
    end = pd.to_datetime(end_date_str, format="%m-%d-%Y").date()
    
    mask_hrv = (df_hrv['date'] >= start) & (df_hrv['date'] <= end)
    period_hrv = df_hrv.loc[mask_hrv].copy()
    
    # Initialize defaults
    raw_stats = {
        "avg_morn": 0, "cv_morn": 0, "avg_rhr": 0, "avg_deep": 0, "avg_core": 0,
        "workout_count": 0, "avg_workout_dur": 0, "avg_recovery": 0
    }
    
    if not period_hrv.empty:
        clean_hrv = period_hrv[(~period_hrv['is_moving']) & (~period_hrv['is_workout_window'])]
        morning_readings = clean_hrv.groupby('date').first()
        
        if not morning_readings.empty:
            raw_stats["avg_morn"] = morning_readings['rMSSD_ms'].mean()
            if len(morning_readings) > 1 and raw_stats["avg_morn"] > 0:
                raw_stats["cv_morn"] = (morning_readings['rMSSD_ms'].std() / raw_stats["avg_morn"]) * 100
                
            if 'hr' in morning_readings.columns:
                raw_stats["avg_rhr"] = morning_readings['hr'].mean()
            elif 'val' in morning_readings.columns:
                raw_stats["avg_rhr"] = morning_readings['val'].mean()

        raw_stats["avg_deep"] = period_hrv[period_hrv['is_deep_sleep']]['rMSSD_ms'].mean()
        raw_stats["avg_core"] = period_hrv[period_hrv['is_sleeping']]['rMSSD_ms'].mean()
        
        period_hrv['day_of_week'] = period_hrv['timestamp'].dt.day_name()
        dow_avgs = period_hrv.groupby('day_of_week')['rMSSD_ms'].mean().round(1).to_dict()
        
        bins = [0, 6, 12, 18, 24]
        labels = ['Night(00-06)', 'Morning(06-12)', 'Afternoon(12-18)', 'Evening(18-24)']
        period_hrv['time_of_day'] = pd.cut(period_hrv['timestamp'].dt.hour, bins=bins, labels=labels, right=False)
        tod_avgs = period_hrv.groupby('time_of_day')['rMSSD_ms'].mean().round(1).to_dict()

    mask_work = (df_workouts['date'] >= start) & (df_workouts['date'] <= end)
    period_workouts = df_workouts.loc[mask_work]
    
    raw_stats["workout_count"] = len(period_workouts)
    if raw_stats["workout_count"] > 0:
        raw_stats["avg_workout_dur"] = period_workouts['duration_min'].mean()
        raw_stats["avg_recovery"] = period_workouts['recovery_time_min'].mean()

    report = f"""
    PERIOD: {start_date_str} to {end_date_str}
    - Morning Readiness (rMSSD): {raw_stats["avg_morn"]:.1f}ms (CV: {raw_stats["cv_morn"]:.1f}%)
    - Resting Heart Rate (Morning): {raw_stats["avg_rhr"]:.1f} bpm
    - Deep Sleep HRV: {raw_stats["avg_deep"]:.1f}ms
    - Core Sleep HRV: {raw_stats["avg_core"]:.1f}ms
    - Day of Week Averages: {dow_avgs if not period_hrv.empty else 'N/A'}
    - Time of Day Averages: {tod_avgs if not period_hrv.empty else 'N/A'}
    - Total Workouts: {raw_stats["workout_count"]}
    - Avg Workout Duration: {raw_stats["avg_workout_dur"]:.1f} min
    - Avg Post-Workout Recovery Time (<100bpm): {raw_stats["avg_recovery"]:.1f} min
    """
    return report, raw_stats

def format_percentage_diff(metric_name, val1, val2, unit=""):
    """Calculates and formats the percentage shift between two values."""
    if val1 == 0 or pd.isna(val1) or pd.isna(val2):
        return f"- {metric_name}: N/A (Missing data)"
    
    pct_change = ((val2 - val1) / val1) * 100
    direction = "Increased" if pct_change > 0 else "Decreased"
    return f"- {metric_name}: {direction} by {abs(pct_change):.2f}% (from {val1:.1f}{unit} to {val2:.1f}{unit})"

# --- PART 2: INITIALIZATION & MAIN LOOP ---

async def main():
    global df_hrv, df_workouts, available_hrv_columns, available_workout_columns
    
    print("=== Neurovis Local Intelligence Engine ===")
    
    load_choice = input("Do you want to load a JSON export for analysis?\nyes/no\n").strip().lower()
    if load_choice not in ['yes', 'y']:
        print("Engine offline.")
        return

    file_path = input("\nprovide file path:\n").strip()
    
    if not os.path.exists(file_path):
        print(f"[!] File not found: {file_path}")
        return

    print("📊 Loading biometric data into Pandas DataFrames...")
    with open(file_path, 'r') as f:
        payload = json.load(f)
        data = payload.get('data', payload)
        
        df_hrv = pd.DataFrame(data.get('hrv', []))
        df_workouts = pd.DataFrame(data.get('workouts', []))
        
        if not df_hrv.empty:
            df_hrv['timestamp'] = pd.to_datetime(df_hrv['timestamp_utc'])
            df_hrv['date'] = df_hrv['timestamp'].dt.date
            available_hrv_columns = list(df_hrv.columns)
            
        if not df_workouts.empty:
            df_workouts['timestamp'] = pd.to_datetime(df_workouts['start_utc'])
            df_workouts['date'] = df_workouts['timestamp'].dt.date
            available_workout_columns = list(df_workouts.columns)

    analyze_biometric_data.__doc__ = f"""
    Analyzes biometric data.
    Args:
        metric (str): MUST be one of {available_hrv_columns} OR {available_workout_columns}
        analysis_type (str): MUST be one of: 'count', 'highest_date', 'lowest_date', 'overall_max', 'overall_min', 'overall_average', 'correlation', 'trend_slope', 'day_of_week_average'
        metric2 (str, optional): Required for correlation.
        start_date (str, optional): The start date for filtering data, format YYYY-MM-DD.
        end_date (str, optional): The end date for filtering data, format YYYY-MM-DD.
    """

    chart_tool = FunctionTool.from_defaults(fn=generate_biometric_chart)
    data_tool = FunctionTool.from_defaults(fn=analyze_biometric_data)
    stats_tool = FunctionTool.from_defaults(fn=calculate_statistical_significance)

    print("🧠 AI Initialized...")
    llm = Ollama(model="qwen2.5", request_timeout=180.0)
    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    neurovis_system_prompt = """
    You are the Neurovis Data Analyst, a highly specialized AI running locally.
    CRITICAL RULES:
    1. You DO NOT write raw Python or Pandas code. You must ONLY use the tools provided to you.
    2. If the user asks for a chart or graph, use the `generate_biometric_chart` tool.
    3. If the user asks if something is statistically significant, use the `calculate_statistical_significance` tool.
    
    DOMAIN GLOSSARY:
    - "HRV" -> 'rMSSD_ms'
    - "Heart Rate" / "Resting HR" -> 'hr' or 'val'
    - "Workout Duration" -> 'duration_min'
    """

    print("🚦 Setting up the ReAct Agent...")
    agent = ReActAgent(
        tools=[data_tool, chart_tool, stats_tool],
        llm=llm,
        system_prompt=neurovis_system_prompt,
        verbose=True,
        max_iterations=10
    )

    chat_history = [] 

    analyze_choice = input("\nDo you want to do deep analysis between two date ranges? (yes/no):\n").strip().lower()
    
    if analyze_choice in ['yes', 'y']:
        try:
            range1 = input("\nAsk start range (MM-DD-YYYY MM-DD-YYYY):\n").strip()
            range2 = input("\nAsk end range (MM-DD-YYYY MM-DD-YYYY):\n").strip()
            
            r1_start, r1_end = range1.split()
            r2_start, r2_end = range2.split()
            
            # Reformat to YYYY-MM-DD for standardizing API/tool calls
            r1_s_fmt = pd.to_datetime(r1_start).strftime("%Y-%m-%d")
            r1_e_fmt = pd.to_datetime(r1_end).strftime("%Y-%m-%d")
            r2_s_fmt = pd.to_datetime(r2_start).strftime("%Y-%m-%d")
            r2_e_fmt = pd.to_datetime(r2_end).strftime("%Y-%m-%d")

            # --- INJECT MEMORY INTO AGENT ---
            memory_injection = f"CRITICAL CONTEXT: The user is currently analyzing Period 1 ({r1_s_fmt} to {r1_e_fmt}) and Period 2 ({r2_s_fmt} to {r2_e_fmt}). If they ask to compare these periods, YOU MUST use these exact YYYY-MM-DD dates in your tool parameters."
            chat_history.append(ChatMessage(role="system", content=memory_injection))

            stats1_report, raw1 = get_deep_period_stats(r1_start, r1_end)
            stats2_report, raw2 = get_deep_period_stats(r2_start, r2_end)
            
            # --- CALCULATE ALL PERCENTAGE DIFFERENTIALS ---
            differentials_block = "\n".join([
                format_percentage_diff("Morning Readiness (rMSSD)", raw1['avg_morn'], raw2['avg_morn'], "ms"),
                format_percentage_diff("Baseline Stability (CV)", raw1['cv_morn'], raw2['cv_morn'], "%"),
                format_percentage_diff("Resting Heart Rate", raw1['avg_rhr'], raw2['avg_rhr'], " bpm"),
                format_percentage_diff("Deep Sleep HRV", raw1['avg_deep'], raw2['avg_deep'], "ms"),
                format_percentage_diff("Core Sleep HRV", raw1['avg_core'], raw2['avg_core'], "ms"),
                format_percentage_diff("Total Workout Count", raw1['workout_count'], raw2['workout_count'], ""),
                format_percentage_diff("Avg Workout Duration", raw1['avg_workout_dur'], raw2['avg_workout_dur'], " min"),
                format_percentage_diff("Post-Workout Recovery Time", raw1['avg_recovery'], raw2['avg_recovery'], " min")
            ])

            # 1. Deep Sleep p-value
            ds1 = df_hrv[(df_hrv['date'] >= pd.to_datetime(r1_start).date()) & (df_hrv['date'] <= pd.to_datetime(r1_end).date()) & (df_hrv['is_deep_sleep'])]['rMSSD_ms'].dropna()
            ds2 = df_hrv[(df_hrv['date'] >= pd.to_datetime(r2_start).date()) & (df_hrv['date'] <= pd.to_datetime(r2_end).date()) & (df_hrv['is_deep_sleep'])]['rMSSD_ms'].dropna()
            _, p_deep = stats.ttest_ind(ds1, ds2, equal_var=False) if len(ds1) > 1 and len(ds2) > 1 else (0, 1)

            # 2. Recovery Time p-value
            rt1 = df_workouts[(df_workouts['date'] >= pd.to_datetime(r1_start).date()) & (df_workouts['date'] <= pd.to_datetime(r1_end).date())]['recovery_time_min'].dropna()
            rt2 = df_workouts[(df_workouts['date'] >= pd.to_datetime(r2_start).date()) & (df_workouts['date'] <= pd.to_datetime(r2_end).date())]['recovery_time_min'].dropna()
            _, p_rec = stats.ttest_ind(rt1, rt2, equal_var=False) if len(rt1) > 1 and len(rt2) > 1 else (0, 1)
            
            analysis_prompt = f"""
            You are a clinical neuro-analyst. I am performing a 21-day intervention to improve my autonomic nervous system. 
            
            {stats1_report}
            
            {stats2_report}

            ABSOLUTE PERCENTAGE SHIFTS (Period 1 to Period 2):
            {differentials_block}

            STATISTICAL SIGNIFICANCE (Welch's t-test):
            - The p-value for the shift in Deep Sleep HRV is {p_deep:.4f}.
            - The p-value for the shift in Post-Workout Recovery Time is {p_rec:.4f}.
            
            Perform a highly detailed clinical comparison between these two periods. Specifically address:
            1. Baseline Stability (CV), Morning Readiness, and Resting Heart Rate. Reference the exact percentage shifts.
            2. Parasympathetic recovery during Deep vs. Core sleep. Reference the percentage shifts.
            3. Circadian alignment based on the Time of Day averages.
            4. Behavioral strain based on the Day of Week averages.
            5. Autonomic recovery efficiency based on the Workout metrics. Reference the percentage shifts.
            6. Explicitly state the p-values and what a statistically significant shift means for physiological homeostasis.
            
            Format your response cleanly with markdown headers. Conclude decisively if the intervention improved systemic homeostasis.
            """
            
            print("\n[*] Pumping massive statistical payload to Qwen...\n")
            response = llm.complete(analysis_prompt)
            print("\n================ DEEP ANALYSIS REPORT ================\n")
            print(response.text)
            print("\n=================================================")
            
        except Exception as e:
            print(f"[!] Error processing ranges: {e}")

    print("\n🚀 ReAct Agent Ready! Type 'exit' or 'quit' to stop.")
    
    while True:
        user_question = input("\nAsk Neurovis: ")
        if user_question.lower() in ['exit', 'quit']:
            print("Shutting down...")
            break
            
        try:
            # Softer prompt anchor so the agent utilizes the dates injected into its memory
            anchored_question = f"CONTEXT: Use the chat history to understand references to 'Period 1' or 'Period 2'.\nNEW QUESTION: {user_question}"
            response = await agent.run(user_msg=anchored_question, chat_history=chat_history)
            
            if isinstance(response, dict) and "response" in response:
                final_answer = str(response["response"])
            else:
                final_answer = str(response)
                
            print(f"\n🤖 Answer: {final_answer}")
            
            chat_history.append(ChatMessage(role="user", content=user_question))
            chat_history.append(ChatMessage(role="assistant", content=final_answer))
            
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main())