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

# --- NEW FASTAPI IMPORTS ---
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import sys
import time
from datetime import datetime

# Suppress pandas chained assignment warnings
warnings.filterwarnings('ignore')

# Global DataFrames
df_hrv = pd.DataFrame()
df_workouts = pd.DataFrame()
available_hrv_columns = []
available_workout_columns = []
morning_readings = pd.DataFrame()

# ==========================================
#  AI ENGINE CONFIGURATION (GLOBAL)
# ==========================================
print(" Initializing AI Engine...")
llm = Ollama(model="qwen2.5", request_timeout=360.0) 
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# ==========================================

# --- PART 1: THE ROBUST PARAMETERIZED TOOLS ---

def generate_biometric_chart(metric1: str, metric2: str = None) -> str:
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
        
        return f"Successfully generated a chart and saved it locally as '{filename}'."
        
    except Exception as e:
        return f"Failed to generate chart: {e}"

def analyze_biometric_data(
    metric: str, 
    analysis_type: Literal[
        'count', 'highest_date', 'lowest_date', 
        'overall_max', 'overall_min', 'overall_average', 'correlation', 
        'trend_slope', 'day_of_week_average', 'most_recent', 'list_dates', 'daily_breakdown'
    ],
    metric2: str = None,
    start_date: str = None,
    end_date: str = None,
    condition_col: str = None
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
                s_date = pd.to_datetime(start_date).date()
                e_date = pd.to_datetime(end_date).date()
                working_df = working_df[(working_df['date'] >= s_date) & (working_df['date'] <= e_date)]
                if working_df.empty:
                    return f"Error: No data found between {start_date} and {end_date}."
            except Exception as e:
                return f"Error parsing dates. Ensure format is YYYY-MM-DD. Details: {e}"

        if condition_col:
            if condition_col in working_df.columns:
                working_df = working_df[working_df[condition_col] == True]
                if working_df.empty:
                    return f"Error: No data found where {condition_col} is True."
            else:
                return f"Error: {condition_col} is not a valid condition."

        if analysis_type == 'count':
            return f"There are {len(working_df)} total data points for {metric} in this period."
            
        elif analysis_type == 'correlation':
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

        elif analysis_type == 'daily_breakdown':
            daily_avgs = working_df.groupby('date')[metric].mean().dropna().round(2)
            if daily_avgs.empty:
                return f"No daily data available for {metric}."
            
            breakdown_str = "\n".join([f"- {date}: {val} ms" for date, val in daily_avgs.items()])
            return f"Here is the daily breakdown for {metric} (filtered by {condition_col if condition_col else 'None'}):\n{breakdown_str}"

        elif analysis_type == 'list_dates':
            unique_dates = sorted(working_df['date'].dropna().unique())
            if not unique_dates:
                return f"No dates found in the dataset for {metric}."
            
            date_strs = [d.strftime('%Y-%m-%d') for d in unique_dates]
            return f"There are {len(date_strs)} unique dates with {metric} data. The dates are: {', '.join(date_strs)}."
            
        elif analysis_type == 'most_recent':
            latest_date = working_df['date'].max()
            latest_data = working_df[working_df['date'] == latest_date]
            daily_avg = latest_data[metric].mean()
            return f"The most recent data is from {latest_date}. The average {metric} on that date was {daily_avg:.2f}."
        # --------------------------
            
        else:
            return f"Error: {analysis_type} is not supported."

    except Exception as e:
        return f"A data calculation error occurred: {str(e)}"

def calculate_statistical_significance(
    metric: str, 
    start_date_1: str, end_date_1: str, 
    start_date_2: str, end_date_2: str
) -> str:
    global df_hrv, df_workouts
    try:
        if metric in available_hrv_columns:
            working_df = df_hrv.copy()
        elif metric in available_workout_columns:
            working_df = df_workouts.copy()
        elif metric == 'score': # <--- NEW: Routes Readiness Score
            if morning_readings.empty:
                return "Error: No morning readings data available."
            working_df = morning_readings.copy()
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
    global df_hrv, df_workouts, morning_readings
    
    start = pd.to_datetime(start_date_str, format="%m-%d-%Y").date()
    end = pd.to_datetime(end_date_str, format="%m-%d-%Y").date()
    
    mask_hrv = (df_hrv['date'] >= start) & (df_hrv['date'] <= end)
    period_hrv = df_hrv.loc[mask_hrv].copy()
    
    raw_stats = {
        "avg_morn": 0, "cv_morn": 0, "avg_deep": 0, "avg_core": 0,
        "avg_resting_hr": 0,
        "workout_count": 0, "avg_workout_dur": 0, "avg_recovery": 0
    }
    
    if not period_hrv.empty:
        sleep_hr_df = period_hrv[(period_hrv['is_sleeping']) & (period_hrv['bpm'] > 0)]
        if not sleep_hr_df.empty:
            raw_stats["avg_resting_hr"] = sleep_hr_df['bpm'].mean()
            
        clean_hrv = period_hrv[(~period_hrv['is_moving']) & (~period_hrv['is_workout_window'])]
        
        if not morning_readings.empty:
            period_morn = morning_readings[(morning_readings['date'] >= start) & (morning_readings['date'] <= end)]
            if not period_morn.empty:
                raw_stats["avg_morn"] = period_morn['score'].mean()
                if len(period_morn) > 1 and raw_stats["avg_morn"] > 0:
                    raw_stats["cv_morn"] = (period_morn['score'].std() / raw_stats["avg_morn"]) * 100

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
    - Deep Sleep HRV: {raw_stats["avg_deep"]:.1f}ms
    - Core Sleep HRV: {raw_stats["avg_core"]:.1f}ms
    - Avg Resting HR: {raw_stats["avg_resting_hr"]:.1f} bpm
    - Day of Week Averages: {dow_avgs if not period_hrv.empty else 'N/A'}
    - Time of Day Averages: {tod_avgs if not period_hrv.empty else 'N/A'}
    - Total Workouts: {raw_stats["workout_count"]}
    - Avg Workout Duration: {raw_stats["avg_workout_dur"]:.1f} min
    - Avg Post-Workout Recovery Time (<100bpm): {raw_stats["avg_recovery"]:.1f} min
    """
    return report, raw_stats

def format_percentage_diff(metric_name, val1, val2, unit=""):
    if val1 == 0 or pd.isna(val1) or pd.isna(val2):
        return f"- {metric_name}: N/A (Missing data)"
    pct_change = ((val2 - val1) / val1) * 100
    direction = "Increased" if pct_change > 0 else "Decreased"
    return f"- {metric_name}: {direction} by {abs(pct_change):.2f}% (from {val1:.1f}{unit} to {val2:.1f}{unit})"


def run_deep_clinical_analysis(start_date_1: str, end_date_1: str, start_date_2: str, end_date_2: str) -> str:
    """
    Performs a comprehensive clinical comparison and statistical assessment between two date ranges.
    Trigger this tool when the user asks to compare two periods, do a deep analysis, or run a full assessment.
    Input formats must be 'YYYY-MM-DD' or 'MM-DD-YYYY'.
    """
    global df_hrv, df_workouts, morning_readings, llm
    
    try:
        r1_start = pd.to_datetime(start_date_1).strftime("%m-%d-%Y")
        r1_end = pd.to_datetime(end_date_1).strftime("%m-%d-%Y")
        r2_start = pd.to_datetime(start_date_2).strftime("%m-%d-%Y")
        r2_end = pd.to_datetime(end_date_2).strftime("%m-%d-%Y")

        stats1_report, raw1 = get_deep_period_stats(r1_start, r1_end)
        stats2_report, raw2 = get_deep_period_stats(r2_start, r2_end)
        
        differentials_block = "\n".join([
            format_percentage_diff("Morning Readiness Score", raw1['avg_morn'], raw2['avg_morn'], ""),
            format_percentage_diff("Baseline Stability (CV)", raw1['cv_morn'], raw2['cv_morn'], "%"),
            format_percentage_diff("Deep Sleep HRV", raw1['avg_deep'], raw2['avg_deep'], "ms"),
            format_percentage_diff("Core Sleep HRV", raw1['avg_core'], raw2['avg_core'], "ms"),
            format_percentage_diff("Average Resting HR", raw1['avg_resting_hr'], raw2['avg_resting_hr'], " bpm"),
            format_percentage_diff("Total Workout Count", raw1['workout_count'], raw2['workout_count'], ""),
            format_percentage_diff("Avg Workout Duration", raw1['avg_workout_dur'], raw2['avg_workout_dur'], " min"),
            format_percentage_diff("Post-Workout Recovery Time", raw1['avg_recovery'], raw2['avg_recovery'], " min")
        ])

        ds1 = df_hrv[(df_hrv['date'] >= pd.to_datetime(r1_start).date()) & (df_hrv['date'] <= pd.to_datetime(r1_end).date()) & (df_hrv['is_deep_sleep'])]['rMSSD_ms'].dropna()
        ds2 = df_hrv[(df_hrv['date'] >= pd.to_datetime(r2_start).date()) & (df_hrv['date'] <= pd.to_datetime(r2_end).date()) & (df_hrv['is_deep_sleep'])]['rMSSD_ms'].dropna()
        _, p_deep = stats.ttest_ind(ds1, ds2, equal_var=False) if len(ds1) > 1 and len(ds2) > 1 else (0, 1)

        rt1 = df_workouts[(df_workouts['date'] >= pd.to_datetime(r1_start).date()) & (df_workouts['date'] <= pd.to_datetime(r1_end).date())]['recovery_time_min'].dropna()
        rt2 = df_workouts[(df_workouts['date'] >= pd.to_datetime(r2_start).date()) & (df_workouts['date'] <= pd.to_datetime(r2_end).date())]['recovery_time_min'].dropna()
        _, p_rec = stats.ttest_ind(rt1, rt2, equal_var=False) if len(rt1) > 1 and len(rt2) > 1 else (0, 1)

        hr1 = df_hrv[(df_hrv['date'] >= pd.to_datetime(r1_start).date()) & (df_hrv['date'] <= pd.to_datetime(r1_end).date()) & (df_hrv['is_sleeping']) & (df_hrv['bpm'] > 0)]['bpm'].dropna()
        hr2 = df_hrv[(df_hrv['date'] >= pd.to_datetime(r2_start).date()) & (df_hrv['date'] <= pd.to_datetime(r2_end).date()) & (df_hrv['is_sleeping']) & (df_hrv['bpm'] > 0)]['bpm'].dropna()
        _, p_hr = stats.ttest_ind(hr1, hr2, equal_var=False) if len(hr1) > 1 and len(hr2) > 1 else (0, 1)
        
        analysis_prompt = f"""
        You are a clinical neuro-analyst. I am performing an intervention to improve my autonomic nervous system. 
        
        {stats1_report}
        
        {stats2_report}

        ABSOLUTE PERCENTAGE SHIFTS (Period 1 to Period 2):
        {differentials_block}

        STATISTICAL SIGNIFICANCE (Welch's t-test):
        - The p-value for the shift in Deep Sleep HRV is {p_deep:.4f}.
        - The p-value for the shift in Post-Workout Recovery Time is {p_rec:.4f}.
        - The p-value for the shift in Resting Heart Rate is {p_hr:.4f}.
        
        CRITICAL GUARDRAIL: If Morning Readiness Score is 0 or "Missing data", you MUST explicitly state this is because "the 30-day rolling baseline requires 30 days of historical data to calculate mathematically," and NOT because the user failed to wear the device.

        Perform a highly detailed clinical comparison between these two periods. Specifically address:
        1. Baseline Stability (CV), Morning Readiness, and Resting HR. Reference the exact percentage shifts.
        2. Parasympathetic recovery during Deep vs. Core sleep. Reference the percentage shifts.
        3. Circadian alignment based on the Time of Day averages.
        4. Behavioral strain based on the Day of Week averages.
        5. Autonomic recovery efficiency based on the Workout metrics. Reference the percentage shifts.
        6. Explicitly state the p-values and what a statistically significant shift means for physiological homeostasis.
        
        Format your response cleanly with markdown headers. Conclude decisively if the intervention improved systemic homeostasis.
        """
        
        print(f"\n[*] Pumping massive statistical payload to Qwen for {start_date_1} vs {start_date_2}...")
        response = llm.complete(analysis_prompt)
        return f"================ DEEP ANALYSIS REPORT ================\n\n{response.text}\n\n================================================="
        
    except Exception as e: 
        return f"[!] Error processing ranges: {e}"

def run_lifestyle_correlations() -> str:
    """
    Analyzes the entire dataset to find mathematical correlations between lifestyle habits...
    """
    global df_hrv, df_workouts
    try:
        # 1. Get Daily Deep Sleep HRV
        deep_sleep = df_hrv[df_hrv['is_deep_sleep'] == True]
        daily_hrv = deep_sleep.groupby('date')['rMSSD_ms'].mean().reset_index()
        daily_hrv.rename(columns={'rMSSD_ms': 'deep_sleep_hrv'}, inplace=True)

        # 2. Get Daily Resting HR
        sleeping_hr = df_hrv[(df_hrv['is_sleeping'] == True) & (df_hrv['bpm'] > 0)]
        daily_rhr = sleeping_hr.groupby('date')['bpm'].mean().reset_index()
        daily_rhr.rename(columns={'bpm': 'resting_hr'}, inplace=True)

        # 3. Get Daily Workout Volume
        daily_workouts = df_workouts.groupby('date').agg(
            total_workout_min=('duration_min', 'sum'),
            avg_recovery_min=('recovery_time_min', 'mean')
        ).reset_index()

        # 4. Merge everything safely by date
        merged = pd.merge(daily_hrv, daily_workouts, on='date', how='inner')
        merged = pd.merge(merged, daily_rhr, on='date', how='inner')
        
        # --- NEW: LAGGED DATA FOR "NEXT DAY" PREDICTIONS ---
        # Sort by date sequentially, then shift the HRV down by 1 row
        merged = merged.sort_values('date')
        merged['next_day_hrv'] = merged['deep_sleep_hrv'].shift(-1)
        
        # Drop the final row since we don't have the "next day" for it yet
        lagged_df = merged.dropna(subset=['next_day_hrv'])

        if len(lagged_df) < 3:
            return "Error: Not enough overlapping days to run a lagged statistical correlation."

        # 5. Calculate Pearson Correlations (Same Day vs Next Day)
        corr_workout_same = merged['total_workout_min'].corr(merged['deep_sleep_hrv'])
        corr_workout_next = lagged_df['total_workout_min'].corr(lagged_df['next_day_hrv'])
        
        corr_recovery_next = lagged_df['avg_recovery_min'].corr(lagged_df['next_day_hrv'])
        corr_rhr_next = lagged_df['resting_hr'].corr(lagged_df['next_day_hrv'])

        report = f"""
        LIFESTYLE CORRELATION MATRIX (Pearson Coefficient -1.0 to 1.0):
        
        SAME-DAY IMPACT (Immediate nervous system response):
        - Total Workout Duration vs Same-Day Deep Sleep HRV: {corr_workout_same:.3f}
        
        NEXT-DAY IMPACT (Delayed recovery/adaptation):
        - Total Workout Duration vs NEXT-DAY Deep Sleep HRV: {corr_workout_next:.3f}
        - Post-Workout Recovery Time vs NEXT-DAY Deep Sleep HRV: {corr_recovery_next:.3f}
        - Resting Heart Rate vs NEXT-DAY Deep Sleep HRV: {corr_rhr_next:.3f}

        HOW TO INTERPRET:
        * +0.5 to +1.0: Strong positive impact.
        * -0.5 to -1.0: Strong negative impact.
        * -0.3 to +0.3: Weak or no mathematical relationship.
        """
        return report
        
    except Exception as e:
        return f"Failed to run correlation engine: {e}"
        

# --- PART 2: API SERVER & LOGGING SETUP ---

app = FastAPI()
agent = None
chat_history = []

class ChatRequest(BaseModel):
    message: str

# Intercepts the verbose ReAct logs and writes them to a file
class LoggerWriter:
    def __init__(self, filename):
        self.file = open(filename, 'a', encoding='utf-8')
    def write(self, message):
        self.file.write(message)
        self.file.flush()
    def flush(self):
        pass

@app.on_event("startup")
async def startup_event():
    global agent
    
    # 1. Initialize Tools
    chart_tool = FunctionTool.from_defaults(fn=generate_biometric_chart)
    data_tool = FunctionTool.from_defaults(fn=analyze_biometric_data)
    stats_tool = FunctionTool.from_defaults(fn=calculate_statistical_significance)
    deep_analysis_tool = FunctionTool.from_defaults(fn=run_deep_clinical_analysis)
    lifestyle_tool = FunctionTool.from_defaults(fn=run_lifestyle_correlations)

    # 2. Define System Prompt
    neurovis_system_prompt = """
    You are the Neurovis Data Analyst, a highly specialized AI running locally.
    CRITICAL RULES:
    1. You DO NOT write raw Python or Pandas code. You must ONLY use the tools provided to you.
    2. If the user asks for a chart or graph, use the `generate_biometric_chart` tool.
    3. If the user asks if something is statistically significant, use the `calculate_statistical_significance` tool.
    4. If the user asks for a deep clinical comparison between two dates, use the `run_deep_clinical_analysis` tool.
    5. STRICT METRIC GUARDRAIL: If the user asks about a metric NOT in the glossary (like Blood Pressure, VO2 Max, or SpO2), you MUST explicitly refuse and state that Neurovis does not track that metric. DO NOT confuse 'bpm' (heart rate) with blood pressure.
    6. STRICT DATA GUARDRAIL: If a tool returns 0, NaN, or "missing data" for a requested time period, you MUST explicitly state that no data exists for those dates.
    7. STRICT DATE GUARDRAIL: If the user asks for data for "each date", "every day", or a "daily breakdown", you MUST use the `daily_breakdown` analysis_type. DO NOT use `day_of_week_average` unless they specifically ask for Monday-Sunday averages.
    8. THE "WHAT WORKS" GUARDRAIL: If the user asks "what works for me", "what affects my HRV", or asks to find trends/correlations between lifestyle habits and recovery, you MUST immediately use the `run_lifestyle_correlations` tool. Do not attempt to calculate these manually.

    DOMAIN GLOSSARY:
    - "HRV" -> 'rMSSD_ms'
    - "Heart Rate" -> 'bpm'
    - "Workout Duration" -> 'duration_min'
    """

    # 3. Setup Agent
    print("Setting up the ReAct Agent...")
    agent = ReActAgent(
        tools=[data_tool, chart_tool, stats_tool, deep_analysis_tool, lifestyle_tool],
        llm=llm,
        system_prompt=neurovis_system_prompt,
        verbose=True, 
        max_iterations=10
    )
    print("Engine Online! Ready for Web Requests. Waiting for Data Upload.")

@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    global df_hrv, df_workouts, morning_readings, available_hrv_columns, available_workout_columns
    
    try:
        contents = await file.read()
        payload = json.loads(contents)
        data = payload.get('data', payload)
        
        df_hrv = pd.DataFrame(data.get('hrv', []))
        df_workouts = pd.DataFrame(data.get('workouts', []))
        morning_readings = pd.DataFrame(payload.get('morning_readings', []))
        
        if not morning_readings.empty:
            morning_readings['date'] = pd.to_datetime(morning_readings['date']).dt.date
            
        if not df_hrv.empty:
            df_hrv['timestamp'] = pd.to_datetime(df_hrv['timestamp_utc'])
            df_hrv['date'] = df_hrv['timestamp'].dt.date
            available_hrv_columns = list(df_hrv.columns)
            
        if not df_workouts.empty:
            df_workouts['timestamp'] = pd.to_datetime(df_workouts['start_utc'])
            df_workouts['date'] = df_workouts['timestamp'].dt.date
            available_workout_columns = list(df_workouts.columns)
            
        return {"message": f"Successfully loaded {len(df_hrv)} HRV records and {len(df_workouts)} workouts."}
        
    except Exception as e:
        return {"message": f"Error loading data: {str(e)}"}

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    try:
        with open("HRVAgent.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Error: index.html not found. Please create it in the same directory.</h1>"

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    global agent, chat_history, available_hrv_columns, available_workout_columns
    
    # 1. Start the stopwatch and grab the timestamp
    start_time = time.time()
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # --- NEW: DYNAMIC SCHEMA INJECTION ---
    # This forces the AI to look at the exact column names before it guesses
    date_range = "Unknown"
    if not df_hrv.empty:
        date_range = f"{df_hrv['date'].min()} to {df_hrv['date'].max()}"

    schema_guide = (
        f"CRITICAL DATA SCHEMA - You may ONLY query these exact column names in your tools:\n"
        f"- df_hrv columns: {available_hrv_columns}\n"
        f"- df_workouts columns: {available_workout_columns}\n"
        f"- morning_readings columns: ['score'] (Use this for Morning Readiness)\n"
        f"DATASET TIME BOUNDARY: The loaded data spans from {date_range}. DO NOT query dates outside this range.\n"
        f"If a requested metric is not in this list, look for a logical alternative before refusing."
    )
    
    anchored_question = f"{schema_guide}\n\nCONTEXT: Use the chat history to understand references to 'Period 1' or 'Period 2'.\nNEW QUESTION: {request.message}"
    # -----------------------------------------------------

    # 2. Write the Request Header to the log file explicitly
    with open("neurovis_agent_thoughts.log", "a", encoding="utf-8") as f:
        f.write(f"\n\n{'='*70}\n")
        f.write(f"[{timestamp_str}] NEW REQUEST\n")
        f.write(f"USER PROMPT: {request.message}\n")
        f.write(f"{'-'*70}\nAGENT THOUGHT PROCESS:\n")
    
    original_stdout = sys.stdout
    sys.stdout = LoggerWriter("neurovis_agent_thoughts.log")
    
    try:
        response = await agent.run(user_msg=anchored_question, chat_history=chat_history)
        
        if isinstance(response, dict) and "response" in response:
            final_answer = str(response["response"])
        else:
            final_answer = str(response)
            
        chat_history.append(ChatMessage(role="user", content=request.message))
        chat_history.append(ChatMessage(role="assistant", content=final_answer))
        
        # 3. Stop the stopwatch
        elapsed_time = time.time() - start_time
        
        # 4. Restore stdout to write the Response Footer explicitly
        sys.stdout = original_stdout
        with open("neurovis_agent_thoughts.log", "a", encoding="utf-8") as f:
            f.write(f"\n{'-'*70}\n")
            f.write(f"FINAL RESPONSE (Generated in {elapsed_time:.2f} seconds):\n{final_answer}\n")
            f.write(f"{'='*70}\n")
            
        return {"reply": final_answer}
        
    except Exception as e:
        sys.stdout = original_stdout
        elapsed_time = time.time() - start_time
        with open("neurovis_agent_thoughts.log", "a", encoding="utf-8") as f:
            f.write(f"\n{'-'*70}\n")
            f.write(f"CRITICAL ERROR (Failed after {elapsed_time:.2f} seconds):\n{str(e)}\n")
            f.write(f"{'='*70}\n")
            
        return {"reply": f"An internal error occurred: {str(e)}"}
        
    finally:
        sys.stdout = original_stdout

if __name__ == "__main__":
    print("\n=======================================================")
    print("STARTING NEUROVIS WEB SERVER")
    print("Open your browser and navigate to: http://localhost:8000")
    print("=======================================================\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)