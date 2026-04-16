import asyncio
import json
import math
import numpy as np
import pandas as pd
import os
import sys
import random
import time
from datetime import datetime
from collections import deque, Counter
import hardware

def ts():
    """Returns a clean timestamp for clinical log syncing"""
    return datetime.now().strftime('%H:%M:%S.%f')[:-3]

# ==========================================
# 1. CONFIGURATION
# ==========================================
SESSIONS_DIR = "research_data"
CONFIG_FILE = "neurovis_config.json"
HISTORY_FILE = "session_history.json"

# Tuning Defaults (Global Starting Points)
MAX_FAA = 1.0         # Max Frontal Alpha Asymmetry value for normalization
MAX_RATIO = 3.0       # Max Beta/Alpha ratio for normalization
BASE_HRV = 50.0       # Baseline HRV (starts at 50, auto-drifts unless locked)
RESTING_HR = 60       # Baseline HR
FAA_OFFSET = 0.0      # Calibration offset for Valence
AX_OFFSET = 0.0       # Minute 1 Null for Anxiety Score

# AXX Baseline Variables
NULL_R_BETA = 0.0     # Minute 1 Null for Cognitive Arousal
NULL_HRV = 50.0       # Minute 1 Null for Somatic Arousal
AXX_OFFSET = 0.0      # Minute 1 Null for Composite Anxiety

BRAIN_WEIGHT = 0.50   # Mixing factor (0.0 = All Body, 1.0 = All Brain)
SMOOTH_FACTOR = 0.90  # Smoothing for Valence (Alpha)
ZONE_WINDOW_SEC = 15  # Window for Zone determination

# Timing Constants (FORENSIC ARCHITECTURE)
RECORD_INTERVAL = 0.05    # 20Hz Recording
BROADCAST_INTERVAL = 0.2  # 5Hz UI Updates

# State Flags (Saved to Config)
is_hrv_locked = False 
hr_source = "POLAR" 

# Note: In a pure browser environment, local file writes might need to use 
# JS hooks, but Pyodide provides a virtual in-memory file system.
if not os.path.exists(SESSIONS_DIR): os.makedirs(SESSIONS_DIR)

# ==========================================
# 2. PERSISTENCE LAYER
# ==========================================
def load_config():
    """ Loads persistent calibration settings from disk. """
    global BASE_HRV, RESTING_HR, FAA_OFFSET, AX_OFFSET, NULL_R_BETA, NULL_HRV, AXX_OFFSET, BRAIN_WEIGHT, is_hrv_locked, hr_source
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                cfg = json.load(f)
                BASE_HRV = float(cfg.get("base_hrv", 50.0))
                RESTING_HR = int(cfg.get("resting_hr", 60))
                FAA_OFFSET = float(cfg.get("faa_offset", 0.0))
                AX_OFFSET = float(cfg.get("ax_offset", 0.0))
                NULL_R_BETA = float(cfg.get("null_r_beta", 0.0))
                NULL_HRV = float(cfg.get("null_hrv", 50.0))
                AXX_OFFSET = float(cfg.get("axx_offset", 0.0))
                BRAIN_WEIGHT = float(cfg.get("brain_weight", 0.50))
                is_hrv_locked = bool(cfg.get("is_hrv_locked", False))
                hr_source = str(cfg.get("hr_source", "POLAR"))
                print(f"[CONFIG] LOADED: BASE={BASE_HRV:.1f}, LOCKED={is_hrv_locked}")
    except Exception as e: 
        print(f"[ERROR] CONFIG LOAD: {e}")

def save_config():
    """ Saves current calibration (Baseline, Offset, Lock Status) to disk. """
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump({
                "base_hrv": BASE_HRV, 
                "resting_hr": RESTING_HR, 
                "faa_offset": FAA_OFFSET,
                "ax_offset": AX_OFFSET,
                "null_r_beta": NULL_R_BETA,
                "null_hrv": NULL_HRV,
                "axx_offset": AXX_OFFSET,
                "brain_weight": BRAIN_WEIGHT,
                "is_hrv_locked": is_hrv_locked,
                "hr_source": hr_source
            }, f, indent=4)
        print(f"[CONFIG] SAVED: BASE={BASE_HRV:.1f}, OFFSET={FAA_OFFSET:.2f}")
    except Exception as e: 
        print(f"[ERROR] CONFIG SAVE: {e}")

def save_session_history(summary_data):
    """ Appends clinical session summary to a master JSON file for long-term trending. """
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
        except: pass
    
    history.append(summary_data)
    
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"[HISTORY] Session logged to {HISTORY_FILE}")
    except Exception as e:
        print(f"[ERROR] HISTORY SAVE: {e}")

# ==========================================
# 3. STATE MANAGEMENT (MULTI-TENANT WRAPPER)
# ==========================================
active_sessions = {}

class UserSession:
    def __init__(self, session_id):
        self.session_id = session_id
        # Replaced ws socket with an outbound message queue for Pyodide UI bridge
        self.outbound_messages = []
        self.hw = hardware.NeuroHardware()
        self.hw.configure_baselines(RESTING_HR, BASE_HRV) 
        
        # Moving all global state tracking into the isolated session
        self.state = {
            "valence": 0.0, "arousal": 0.0, "zone": "NEUTRAL",
            "hr": 0, "rmssd": 0, "sdnn": 0,
            "faa_raw": 0.0, "ratio_raw": 0.0,
            "smooth_L": 0.0, "smooth_R": 0.0,
            "smooth_ratio": 0.0, 
            "annotation": "",
            "stimulus_active": False, 
            "phase_label": "",
            
            # --- NEW: ARTIFACT TRACKING ---
            "artifact_active": False,
            "artifact_start_time": 0.0,
            "phase_override": None,
            "last_good_L_alpha": 0.0,
            "last_good_R_alpha": 0.0,
            "last_good_L_beta": 0.0,
            "last_good_R_beta": 0.0
        }

        # Chaos Test Data Buckets (For Scoring)
        self.chaos_phase_data = { "BASELINE": [], "ANTICIPATION": [], "STRESS": [], "RECOVERY": [] }

        # Buffers
        self.rr_buffer = deque(maxlen=30)      # Rolling buffer for RMSSD calc
        self.hrv_history = deque(maxlen=300)   # History for Auto-Drift baseline
        self.x_buffer = deque(maxlen=30)       # Smoothing buffer for X-Axis
        self.y_buffer = deque(maxlen=30)       # Smoothing buffer for Y-Axis
        self.zone_buffer_x = deque(maxlen=int(ZONE_WINDOW_SEC * 5))
        self.zone_buffer_y = deque(maxlen=int(ZONE_WINDOW_SEC * 5))
        self.muse_1min_rr_buffer = []
        self.muse_hrv_window_start = datetime.now()

        # Session Metrics
        self.session_data = []
        self.zone_counts = Counter()
        self.total_samples = 0
        self.is_recording = False
        self.is_calibrating = False

        # Calibration Buffers
        self.calib_buffer_faa = []
        self.calib_buffer_hrv = []
        self.calib_buffer_hr = []

        # Filter State Globals (Clean Room Logic)
        self.consecutive_rejections = 0
        self.last_valid_beat_time = datetime.now()

        # Loop Timers
        self.last_record_time = 0
        self.last_broadcast_time = 0
        
        # AX Ticker Globals
        self.last_ax_tick = time.time()
        self.ax_minute_buffer = []
        self.axx_minute_buffer = []
        self.smooth_val = 0.0
        self.smooth_aro = 0.0

# ==========================================
# 4. MATH & LOGIC
# ==========================================
def calculate_metrics(session, hw_data):
    """
    Forensic Filtering Logic:
    1. 'Clean Room' Buffer: Rejects artifacts (jumps/loose sensor) to prevent false spikes.
    2. 'Staleness Check': Flushes buffer if no valid beats occur for 3s (prevents flatlining).
    3. 'Display Gate': Forces RMSSD to 0 if variability implies physically impossible heart movement (artifacts).
    """
    # ----------------------------------------
    # 1. STALENESS CHECK (The Auto-Flush)
    # ----------------------------------------
    time_since_last = (datetime.now() - session.last_valid_beat_time).total_seconds()
    
    # ----------------------------------------
    # 2 & 3. INPUT PROCESSING & OUTPUT
    # ----------------------------------------
    if hr_source == "POLAR":
        # Flushes ONLY the Polar buffer if stale
        if time_since_last > 3.0:
            if len(session.rr_buffer) > 0:
                session.rr_buffer.clear()
            session.state["rmssd"] = 0 
            session.state["sdnn"] = 0

        incoming_rr = hw_data.get("polar_rr_intervals", [])

        while incoming_rr:
            rr = incoming_rr.pop(0)
            if 200 < rr < 1400:
                if len(session.rr_buffer) > 0:
                    last_rr = session.rr_buffer[-1]
                    delta = abs(rr - last_rr)
                    
                    if delta <= (last_rr * 0.60):
                        session.rr_buffer.append(rr)
                        session.consecutive_rejections = 0 
                        session.last_valid_beat_time = datetime.now() 
                    else:
                        session.consecutive_rejections += 1
                        if session.consecutive_rejections > 3:
                            session.rr_buffer.append(rr) 
                            session.consecutive_rejections = 0
                            session.last_valid_beat_time = datetime.now()
                else:
                    session.rr_buffer.append(rr)
                    session.last_valid_beat_time = datetime.now()
            
        if len(session.rr_buffer) > 10:
            diffs = np.diff(np.array(session.rr_buffer))
            raw_rmssd = np.sqrt(np.mean(diffs ** 2))
            session.state["sdnn"] = np.std(np.array(session.rr_buffer))
            if raw_rmssd > 120: session.state["rmssd"] = 0
            else: session.state["rmssd"] = raw_rmssd
        else:
            if time_since_last > 3.0: session.state["rmssd"] = 0

    elif hr_source == "MUSE":
        incoming_rr = hw_data.get("muse_rr_intervals", [])

        while incoming_rr:
            rr = incoming_rr.pop(0)
            
            if 200 < rr < 1400:
                if len(session.muse_1min_rr_buffer) > 0:
                    last_rr = session.muse_1min_rr_buffer[-1]
                    delta = abs(rr - last_rr)
                    
                    if delta <= (last_rr * 0.60):
                        session.muse_1min_rr_buffer.append(rr)
                        session.consecutive_rejections = 0 
                        session.last_valid_beat_time = datetime.now() 
                        print(f"❤️ [{session.session_id}] [MUSE HR] Beat mapped! Buffer: {len(session.muse_1min_rr_buffer)} beats.")
                    else:
                        session.consecutive_rejections += 1
                        if session.consecutive_rejections > 3:
                            session.muse_1min_rr_buffer.append(rr) 
                            session.consecutive_rejections = 0
                            session.last_valid_beat_time = datetime.now()
                else:
                    session.muse_1min_rr_buffer.append(rr)
                    session.last_valid_beat_time = datetime.now()
                    print(f"❤️ [{session.session_id}] [MUSE HR] First beat mapped! Buffer: 1")

        # The 60-Second Tumbling Window Math
        window_elapsed = (datetime.now() - session.muse_hrv_window_start).total_seconds()
        
        if window_elapsed >= 60.0:
            print(f"⏱️ [{session.session_id}] [MUSE HRV] 60-second window closed. Processing {len(session.muse_1min_rr_buffer)} collected beats...")
            
            if len(session.muse_1min_rr_buffer) > 30: # Need at least 30 clean beats in the minute
                diffs = np.diff(np.array(session.muse_1min_rr_buffer))
                raw_rmssd = np.sqrt(np.mean(diffs ** 2))
                session.state["sdnn"] = np.std(np.array(session.muse_1min_rr_buffer))
                
                if raw_rmssd > 120: 
                    session.state["rmssd"] = 0
                    print(f"❌ [{session.session_id}] [MUSE HRV] REJECTED. RMSSD ({raw_rmssd:.1f}) implies artifact.")
                else: 
                    session.state["rmssd"] = raw_rmssd
                    print(f"✅ [{session.session_id}] [MUSE HRV] SUCCESS! New 1-Min HRV: {session.state['rmssd']:.1f}")
            else:
                print(f"❌ [{session.session_id}] [MUSE HRV] FAILED. Not enough valid beats ({len(session.muse_1min_rr_buffer)}/30).")
                # Keep the previous minute's HRV rather than dropping to 0, which looks like a crash
            
            # Reset for the next minute
            session.muse_1min_rr_buffer.clear()
            session.muse_hrv_window_start = datetime.now()

def determine_zone(val, aro):
    """ Maps Valence/Arousal vectors to named Emotional Zones """
    distance = math.sqrt(val**2 + aro**2)
    if distance < 0.25: return "NEUTRAL"
    if val > 0 and aro > 0: return "ENGAGEMENT"
    if val < 0 and aro > 0: return "VIGILANCE"
    if val > 0 and aro < 0: return "FLOW"
    if val < 0 and aro < 0: return "DETACHMENT"
    return "NEUTRAL"

def normalize_metrics(session, faa, ratio, rmssd, hr):
    """
    Normalizes raw bio-signals into -1.0 to 1.0 vectors.
    Includes 'Auto-Drift' logic to adapt to user's baseline over time.
    """
    global BASE_HRV, FAA_OFFSET, is_hrv_locked, RESTING_HR, BRAIN_WEIGHT
    
    # --- AUTO-DRIFT LOGIC ---
    # Slowly adjusts baseline if not locked, allowing the system to learn the user.
    if not is_hrv_locked and rmssd > 0 and not session.is_calibrating:
        session.hrv_history.append(rmssd)
        if len(session.hrv_history) > 10:
            BASE_HRV = (BASE_HRV * 0.99) + (rmssd * 0.01)

    # 1. Valence (Asymmetry)
    # Right-Frontal Alpha > Left-Frontal Alpha = Approach (Positive)
    corrected_faa = faa - FAA_OFFSET
    val = max(-1.0, min(1.0, corrected_faa / MAX_FAA))
    
    # 2. Arousal (Dynamic Weighting)
    # Combines Brain (Beta/Alpha) and Body (HR + HRV)
    norm_ratio = min(1.0, session.state["smooth_ratio"] / MAX_RATIO)
    
    # Invert HRV (Lower HRV = Higher Stress/Arousal)
    if rmssd == 0: norm_hrv_stress = 0
    else: norm_hrv_stress = max(-1.0, min(1.0, (BASE_HRV - rmssd) / BASE_HRV))
        
    if hr == 0: norm_hr_stress = 0
    else: norm_hr_stress = max(-1.0, min(1.0, (hr - RESTING_HR) / 30.0))

    # Dynamic Split based on Brain Weight slider
    body_weight = 1.0 - BRAIN_WEIGHT
    hr_share = body_weight / 2
    hrv_share = body_weight / 2
    
    aro = (norm_hrv_stress * hrv_share) + (norm_hr_stress * hr_share) + (norm_ratio * BRAIN_WEIGHT)

    return val, max(-1.0, min(1.0, aro))

def calculate_ian_score(session):
    """ Calculates the Relative Resilience Score based on phase drops during Chaos Test. """
    base_vals = [x for x in session.chaos_phase_data["BASELINE"] if x > 0]
    if not base_vals: return 0
    baseline = np.mean(base_vals)

    # 1. Anticipation (Stability)
    anti_vals = [x for x in session.chaos_phase_data["ANTICIPATION"] if x > 0]
    anti_score = 20
    if anti_vals and baseline > 0:
        drop_pct = (baseline - np.mean(anti_vals)) / baseline
        if drop_pct > 0: anti_score -= (drop_pct * 100) * 2
    anti_score = max(0, min(20, anti_score))

    # 2. Stress (Reactivity)
    stress_vals = [x for x in session.chaos_phase_data["STRESS"] if x > 0]
    stress_score = 40
    if stress_vals and baseline > 0:
        drop_pct = (baseline - np.mean(stress_vals)) / baseline
        excess_drop = max(0, drop_pct - 0.20)
        stress_score -= (excess_drop * 100) * 2
    stress_score = max(0, min(40, stress_score))

    # 3. Recovery (Rebound)
    rec_vals = [x for x in session.chaos_phase_data["RECOVERY"] if x > 0]
    rec_score = 0
    if rec_vals and baseline > 0:
        recovery_ratio = np.mean(rec_vals) / baseline
        rec_score = min(40, 40 * recovery_ratio)

    total = int(anti_score + stress_score + rec_score)
    return max(0, min(100, total))

# ==========================================
# 5. AUTOMATED TEST PROTOCOLS
# ==========================================
async def sleep_until(target_time):
    now = time.time()
    delta = target_time - now
    if delta > 0: await asyncio.sleep(delta)

async def run_clinical_calibration(session, phase_label_prefix="SESSION"):
    """
    SHARED CLINICAL CALIBRATION ENGINE
    Standardizes baseline collection across ALL tests.
    1 Minute | Eyes CLOSED | Open Monitoring (True Rest)
    Updates global baseline config upon completion.
    """
    global BASE_HRV, RESTING_HR, is_hrv_locked
    global AX_OFFSET, NULL_R_BETA, NULL_HRV, AXX_OFFSET, FAA_OFFSET
    
    calib_label = f"{phase_label_prefix}_CALIBRATION"
    session.state["phase_label"] = calib_label
    
    print(f"[{session.session_id}] [CALIB] STARTING STANDARD 1-MIN CALIBRATION ({calib_label})")
    
    prompt = "GLOBAL CALIBRATION \nSit comfortably with your eyes CLOSED.\nDo not count or control breath.\nJust notice whatever arises in your mind"
    session.outbound_messages.append({"type": "VALENCE_PROMPT", "val": prompt, "mode": "NEUTRAL"})
    
    # Run for 60 seconds
    await asyncio.sleep(60)
    
    # Process Data and extract Minute 1 Null parameters
    df = pd.DataFrame(session.session_data)
    if not df.empty:
        subset = df[df['manual_label'] == calib_label]
        if not subset.empty:
            if 'rmssd' in subset.columns:
                new_base = subset['rmssd'].mean()
                if new_base > 10: 
                    BASE_HRV = new_base
                    NULL_HRV = new_base
            if 'hr' in subset.columns:
                new_rest = subset['hr'].mean()
                if new_rest > 30: RESTING_HR = int(new_rest)
            if 'raw_ax' in subset.columns:
                AX_OFFSET = subset['raw_ax'].mean()
            if 'R_beta' in subset.columns:
                NULL_R_BETA = subset['R_beta'].mean()
            if 'raw_axx' in subset.columns:
                AXX_OFFSET = subset['raw_axx'].mean()
            if 'raw_faa' in subset.columns:
                FAA_OFFSET = subset['raw_faa'].mean()
            
            print(f"[{session.session_id}] [CALIB] UPDATED BASELINES -> HRV: {BASE_HRV:.1f}, HR: {RESTING_HR}")
            is_hrv_locked = True
            save_config()

            # --- NEW: Push to UI for DB Save ---
            calib_payload = {
                "type": "SAVE_CALIB_DB",
                "base_hrv": BASE_HRV, "resting_hr": RESTING_HR, "faa_offset": FAA_OFFSET,
                "ax_offset": AX_OFFSET, "null_r_beta": NULL_R_BETA, "null_hrv": NULL_HRV, "axx_offset": AXX_OFFSET
            }
            session.outbound_messages.append(calib_payload)
        else:
            print(f"[{session.session_id}] [CALIB FAIL] All data rejected as Artifacts.")
    else:
        print(f"[{session.session_id}] [CALIB FAIL] No data collected. Was the headset worn?")


async def run_unified_session(session):
    """ 
    THE UNIFIED PROTOCOL (15 Minutes)
    Calibration -> Stim -> Chaos -> Valence
    Records to ONE master CSV file.
    """
    print(f"[{session.session_id}] STARTING UNIFIED PROTOCOL")
    session.is_recording = True; session.session_data = []; session.zone_counts = Counter(); session.total_samples = 0
    session.chaos_phase_data = {k:[] for k in session.chaos_phase_data} 
    
    # 1. INTRO
    session.outbound_messages.append({"type": "VALENCE_PROMPT", "val": "WELCOME TO THE UNIFIED ASSESSMENT", "mode": "NEUTRAL"})
    await asyncio.sleep(5)

    # 2. GLOBAL CALIBRATION (1m)
    await run_clinical_calibration(session, "UNIFIED")

    # 3. STIMULUS TEST (2m)
    print(f"[{session.session_id}] [UNIFIED] STIMULUS PHASE")
    prompt = "STIMULUS PHASE \n In this test we are going to test your mental fitness and resilience.\n Please close eyes an relax.\nLet your body react naturally."
    session.outbound_messages.append({"type": "VALENCE_PROMPT", "val": prompt, "mode": "NEUTRAL"})
    
    session.state["phase_label"] = "STIM_PRE"
    await asyncio.sleep(25) 
    
    # Shock 1
    session.state["stimulus_active"] = True; session.state["phase_label"] = "STIM_SHOCK"
    session.outbound_messages.append({"type": "PLAY_STIMULUS"})
    await asyncio.sleep(60) 
    
    # Shock 2
    session.state["stimulus_active"] = True; session.state["phase_label"] = "STIM_SHOCK"
    session.outbound_messages.append({"type": "PLAY_STIMULUS"})
    await asyncio.sleep(60) 
    session.state["phase_label"] = "STIM_RECOVER"

    # 4. TRANSITION (Instructions)
    session.state["phase_label"] = "CHAOS_INSTRUCT"
    prompt = "CHAOS TEST INSTRUCTIONS\nOpen eyes. You will perform a cognitive challenge.\nType the COLOR of the word, or solve math.\nPrepare yourself and focus on speed and accuracy."
    session.outbound_messages.append({"type": "VALENCE_PROMPT", "val": prompt, "mode": "NEUTRAL"})
    await asyncio.sleep(15)

    # 5. CHAOS BASELINE (30s)
    session.state["phase_label"] = "CHAOS_BASE"
    prompt = "VISUAL BASELINE \nStare at the pulsing dot.\nDo not move your eyes, just empty your mind"
    session.outbound_messages.append({"type": "TRAIN_PROMPT", "val": prompt, "mode": "CALIB_VISUAL"})
    await asyncio.sleep(30)

    # --- FIX: CLEAR THE DOT BEFORE STARTING GAME ---
    session.outbound_messages.append({"type": "VALENCE_PROMPT", "val": "", "mode": "NEUTRAL"})
    
    # 6. CHAOS GAME (60s)
    session.state["phase_label"] = "CHAOS_ACTIVE"
    session.outbound_messages.append({"type": "CHAOS_START_GAME"})
    await asyncio.sleep(60)
    session.outbound_messages.append({"type": "CHAOS_STOP_GAME"})

    # 7. BREAK (20s)
    session.state["phase_label"] = "CHAOS_BREAK"
    prompt = "BREAK \nTake a deep breath and clear your mind. That was an intense test meant to generate frustration. Let that go now"
    session.outbound_messages.append({"type": "VALENCE_PROMPT", "val": prompt, "mode": "CALM"})
    await asyncio.sleep(20)

    # 8. VALENCE TEST (3m)
    print(f"[{session.session_id}] [UNIFIED] VALENCE PHASE")
    
    # Positive
    session.state["phase_label"] = "VAL_POS"
    prompt = "VALENCE: POSITIVE\n Now please close your eyes. Visualize your most positive memories, bring to mind things that bring you joy and happyness."
    session.outbound_messages.append({"type": "VALENCE_PROMPT", "val": prompt, "mode": "POSITIVE"})
    await asyncio.sleep(60)

    # Negative
    session.state["phase_label"] = "VAL_NEG"
    prompt = "VALENCE: NEGATIVE\nKeep Eyes Closed. Bring to mind moments of sadness, regret, or anxiety."
    session.outbound_messages.append({"type": "VALENCE_PROMPT", "val": prompt, "mode": "NEGATIVE"})
    await asyncio.sleep(60)

    # Washout
    session.state["phase_label"] = "VAL_WASH"
    prompt = "RECOVERY \nLet that go - that was intense. Focus on your breath and visulize joy."
    session.outbound_messages.append({"type": "VALENCE_PROMPT", "val": prompt, "mode": "CALM"})
    await asyncio.sleep(60)

    # 9. END
    session.is_recording = False
    print(f"[{session.session_id}] UNIFIED PROTOCOL COMPLETE.")
    prompt = "TEST COMPLETE\nOpen your eyes."
    session.outbound_messages.append({"type": "VALENCE_WASHOUT", "val": prompt})

    # Save Master File - Optional in browser, better to hand data back to UI
    fn = f"neurovis_unified_{session.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # GENERATE JSON SUMMARY
    df = pd.DataFrame(session.session_data)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "type": "UNIFIED_ASSESSMENT",
        "session_id": session.session_id,
        "overall": {
            "avg_hrv": float(df['rmssd'].mean()) if not df.empty else 0.0, 
            "avg_ax": float(df['live_ax'].mean()) if not df.empty and 'live_ax' in df.columns else 0.0,
            "avg_axx": float(df['live_axx'].mean()) if not df.empty and 'live_axx' in df.columns else 0.0,
            "duration_sec": len(df) * RECORD_INTERVAL
        }
    }
    
    session.outbound_messages.append({ "type": "TEST_COMPLETE", "filename": fn, "summary": summary })

    # --- NEW: TRIGGER WEB DOWNLOAD & TAGGING MODAL ---
    if session.session_data:
        session.outbound_messages.append({
            "type": "SAVE_SESSION_DATA", 
            "payload": json.dumps(session.session_data)
        })
        session.session_data = [] # Wipe RAM after sending

async def run_training_session(session):
    """ 
    DAILY NEUROVIS SESSION (12 Minutes)
    This is the "Daily Gym" for Vagal Tone training.
    1m Calibration -> 5m Resonant Breathing -> 6m Positivity Training
    """
    print(f"[{session.session_id}] STARTING DAILY NEUROVIS SESSION (12:00)")
    session.is_recording = True; session.session_data = []; session.zone_counts = Counter(); session.total_samples = 0
    
    t_zero = time.time()
    t_res_end = t_zero + 300.0 + 60.0 # +60 for calib time
    t_pos_end = t_res_end + 360.0
    
    # 1. Standard Calibration
    await run_clinical_calibration(session, "SESSION")

    # 2. Resonance
    session.state["phase_label"] = "SESSION_RESONANCE"
    prompt = "RESONANCE TRAINING (5m)\nClose eyes. Inhale 4s, Exhale 6s.\nFocus entirely on the breath."
    session.outbound_messages.append({"type": "TRAIN_PROMPT", "val": prompt, "mode": "BREATH_PACER"})
    
    await sleep_until(t_res_end)
    
    # 3. Positivity
    session.state["phase_label"] = "SESSION_POSITIVITY"
    prompt = "POSITIVE NEUROPLASTICITY (6m)\nKeep eyes closed. Visualize a moment of pure success or joy.\nRaise the yellow line."
    session.outbound_messages.append({"type": "TRAIN_PROMPT", "val": prompt, "mode": "POSITIVE_VIS"})
    
    await sleep_until(t_pos_end)
    
    session.is_recording = False
    print(f"[{session.session_id}] NEUROVIS SESSION COMPLETE.")
    
    prompt = "SESSION COMPLETE\nOpen your eyes."
    session.outbound_messages.append({"type": "VALENCE_WASHOUT", "val": prompt})
    
    # GENERATE JSON SUMMARY
    df = pd.DataFrame(session.session_data)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "type": "DAILY_SESSION_12MIN",
        "session_id": session.session_id,
        "config": {"base_hrv": BASE_HRV, "resting_hr": RESTING_HR},
        "overall": {
            "avg_hrv": float(df['rmssd'].mean()) if not df.empty else 0.0, 
            "avg_ax": float(df['live_ax'].mean()) if not df.empty and 'live_ax' in df.columns else 0.0,
            "avg_axx": float(df['live_axx'].mean()) if not df.empty and 'live_axx' in df.columns else 0.0,
            "duration_sec": len(df) * RECORD_INTERVAL
        }
    }
    
    fn = f"neurovis_session_{session.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    session.outbound_messages.append({ "type": "TEST_COMPLETE", "filename": fn, "summary": summary })

    # --- NEW: TRIGGER WEB DOWNLOAD & TAGGING MODAL ---
    if session.session_data:
        session.outbound_messages.append({
            "type": "SAVE_SESSION_DATA", 
            "payload": json.dumps(session.session_data)
        })
        session.session_data = [] # Wipe RAM after sending

async def run_manual_calibration_button(session):
    """ Triggered by the UI 'Auto-Calibrate' Button. Runs standard logic. """
    # Enable recording for buffer math
    session.is_recording = True; session.session_data = [] 
    
    await run_clinical_calibration(session, "MANUAL")
    
    session.is_recording = False
    # Explicitly signal done to reset UI
    prompt = "Calibration Complete."
    session.outbound_messages.append({"type": "VALENCE_WASHOUT", "val": prompt})
    session.outbound_messages.append({"type": "CALIB_DONE"})

# ==========================================
# 6. THE PYODIDE BRIDGE (Replaces WebSockets & main loop)
# ==========================================
def process_tick(sid, hardware_payload_dict=None, ui_command=None):
    """ 
    This is called directly by JavaScript @ 20Hz. 
    It ingests data, runs the math synchronously, and returns the UI state directly.
    """
    global BASE_HRV, RESTING_HR, is_hrv_locked, BRAIN_WEIGHT, hr_source, FAA_OFFSET, AX_OFFSET, NULL_HRV, NULL_R_BETA, AXX_OFFSET

    # Make sure we've loaded config on the very first tick
    if not active_sessions:
        load_config()
        print("NEUROVIS v4.0 (PYODIDE WASM ENGINE) STARTUP")

    # 1. Initialize Session if new
    if sid not in active_sessions:
        active_sessions[sid] = UserSession(sid)
        
    session = active_sessions[sid]
    now = time.time()

    # 2. Process UI Commands
    if ui_command:
        cmd = ui_command.get("cmd")
        val = ui_command.get("val")

        # --- RECORDING CONTROLS ---
        if cmd == 'START_REC':
            session.is_recording = True
            session.session_data = []
            session.zone_counts = Counter()
            session.total_samples = 0
            session.state["phase_label"] = "MANUAL_REC"
            print(f"[{sid}] REC STARTED")
            
        elif cmd == 'STOP_REC':
            session.is_recording = False
            session.state["phase_label"] = ""
            if session.session_data:
                # Pack as JSON string to bypass Pyodide nested dictionary bugs
                session.outbound_messages.append({
                    "type": "SAVE_SESSION_DATA", 
                    "payload": json.dumps(session.session_data)
                })
                session.session_data = [] # Wipe RAM after sending
                
        elif cmd == 'ANNOTATE': 
            session.state["annotation"] = val
            if val == 'ANXIETY_MARKED':
                async def clear_tag(s):
                    await asyncio.sleep(2)
                    if s.state["annotation"] == 'ANXIETY_MARKED':
                        s.state["annotation"] = ""
                # Pyodide's event loop will run this correctly
                asyncio.create_task(clear_tag(session))
        
        # --- START AUTOMATED TESTS ---
        elif cmd == 'START_UNIFIED':
            asyncio.create_task(run_unified_session(session))
        elif cmd == 'START_TRAINING':
            asyncio.create_task(run_training_session(session))

        # --- LOCKING LOGIC (GLOBAL SETTINGS) ---
        elif cmd == 'SET_BASELINE': 
            BASE_HRV = float(val); is_hrv_locked = True
            save_config()
            session.hw.configure_baselines(RESTING_HR, BASE_HRV)

        elif cmd == 'TOGGLE_HRV_LOCK':
            is_hrv_locked = not is_hrv_locked; save_config()
            
        elif cmd == 'SET_RESTING_HR': 
            RESTING_HR = int(val)
            save_config() 
            session.hw.configure_baselines(RESTING_HR, BASE_HRV)

        elif cmd == 'SET_FAA_OFFSET': 
            FAA_OFFSET = float(val); save_config()
        elif cmd == 'SET_AX_OFFSET': 
            AX_OFFSET = float(val); save_config()
        elif cmd == 'SET_AXX_OFFSET': 
            AXX_OFFSET = float(val); save_config()
        elif cmd == 'SET_NULL_HRV': 
            NULL_HRV = float(val); save_config()
        elif cmd == 'SET_NULL_R_BETA': 
            NULL_R_BETA = float(val); save_config()
        
        elif cmd == 'SET_BRAIN_WEIGHT':
            BRAIN_WEIGHT = float(val); save_config()

        elif cmd == 'SET_HR_SOURCE':
            hr_source = val; save_config()
            print(f"[{sid}] [CONFIG] HR SOURCE SWITCHED TO: {hr_source}")

        elif cmd == 'START_CALIB':
            asyncio.create_task(run_manual_calibration_button(session))

    # 3. Ingest Hardware Data
    if hardware_payload_dict:
        session.hw.process_web_payload(hardware_payload_dict)

    data = session.hw.get_latest() 
    
    # 4. STATUS GATEKEEPER
    if data["quality_flag"] != "OK":
        if now - session.last_broadcast_time > BROADCAST_INTERVAL:
            session.last_broadcast_time = now
            payload = {
                "type": "LIVE", "x": session.smooth_val, "y": session.smooth_aro, "zone": session.state["zone"],
                "hr": session.state["hr"], "rmssd": float(session.state["rmssd"]),
                "raw_faa": session.state["faa_raw"], 
                "base_hrv": BASE_HRV, "resting_hr": RESTING_HR,
                "faa_offset": FAA_OFFSET, "ax_offset": AX_OFFSET, "axx_offset": AXX_OFFSET,
                "null_hrv": NULL_HRV, "null_r_beta": NULL_R_BETA,
                "battery": data.get("battery", 0),
                "muse_batt": data.get("muse_batt", 0),
                "polar_batt": data.get("polar_batt", 0),
                "motion_gate": data.get("motion_gate", False),
                "quality_flag": data["quality_flag"],
                "L_beta": data.get('L_beta'),
                "R_beta": data.get('R_beta')
            }
            if session.outbound_messages:
                payload["alerts"] = session.outbound_messages.copy()
                session.outbound_messages.clear()
            return payload
            
        return None

    # 5. METRICS PROCESSING
    if hr_source == "MUSE": session.state["hr"] = data["muse_hr"]
    else: session.state["hr"] = data["polar_hr"]

    calculate_metrics(session, data) 
    
    # --- THE SYMMETRY FAILSAFE & MOTION GATE ---
    # Temporarily shift up by 6 Bels (making minimum exactly 0.0) just for the check
    chk_L = data["L_alpha"] + 6.0
    chk_R = data["R_alpha"] + 6.0
    
    is_asymmetric = (chk_L > 2.5 * chk_R) or (chk_R > 2.5 * chk_L)
    is_motion = data.get("motion_gate", False)
    
    if is_asymmetric or is_motion:
        if not session.state["artifact_active"]:
            session.state["artifact_active"] = True
            session.state["artifact_start_time"] = now
            reason = "MOTION" if is_motion else "ASYMMETRY"
            print(f"[{ts()}]  [{sid}] {reason} ARTIFACT DETECTED. Engaging Micro-Hold.")
        
        artifact_duration = now - session.state["artifact_start_time"]
        
        if artifact_duration <= 2.0:
            # TIER 1: MICRO-HOLD (< 2 seconds) -> Override with last known good data
            data["L_alpha"] = session.state.get("last_good_L_alpha", data["L_alpha"])
            data["R_alpha"] = session.state.get("last_good_R_alpha", data["R_alpha"])
            data["L_beta"]  = session.state.get("last_good_L_beta", data["L_beta"])
            data["R_beta"]  = session.state.get("last_good_R_beta", data["R_beta"])
        else:
            # TIER 2: GRACEFUL PAUSE (> 2 seconds) -> Sustain hold, but flag the CSV
            if session.state["phase_override"] != "ARTIFACT":
                print(f"[{ts()}] [{sid}] SUSTAINED ARTIFACT (>2s). Suspending math and flagging CSV.")
                session.state["phase_override"] = "ARTIFACT"
            
            data["L_alpha"] = session.state.get("last_good_L_alpha", data["L_alpha"])
            data["R_alpha"] = session.state.get("last_good_R_alpha", data["R_alpha"])
            data["L_beta"]  = session.state.get("last_good_L_beta", data["L_beta"])
            data["R_beta"]  = session.state.get("last_good_R_beta", data["R_beta"])

    else:
        # SENSOR IS CLEAN
        if session.state["artifact_active"]:
            print(f"[{ts()}] [{sid}] ARTIFACT CLEARED. Resuming live telemetry.")
            session.state["artifact_active"] = False
            session.state["phase_override"] = None
        
        # Update our "Last Known Good" memory
        session.state["last_good_L_alpha"] = data["L_alpha"]
        session.state["last_good_R_alpha"] = data["R_alpha"]
        session.state["last_good_L_beta"]  = data["L_beta"]
        session.state["last_good_R_beta"]  = data["R_beta"]

    # SMOOTHING FACTORS (Affective Physics)
    # Valence (Alpha) Smoothing
    session.state["smooth_L"] = (session.state["smooth_L"] * SMOOTH_FACTOR) + (data["L_alpha"] * (1-SMOOTH_FACTOR))
    session.state["smooth_R"] = (session.state["smooth_R"] * SMOOTH_FACTOR) + (data["R_alpha"] * (1-SMOOTH_FACTOR))
    raw_faa = session.state["smooth_R"] - session.state["smooth_L"]
    session.state["faa_raw"] = max(-2.0, min(2.0, raw_faa))
    
    # --- AX & AXX SCORE (ANXIETY) CALCULATION ---
    # 1. Base AX Score (Direction of Emotion)
    threat_scanner = data["R_alpha"] * -1
    problem_solver = data["L_alpha"] * -1
    raw_ax = threat_scanner - problem_solver
    live_ax = raw_ax - AX_OFFSET
    
    # 2. AXX Multipliers (Velocity of Emotion)
    live_r_beta_lin = 10 ** data["R_beta"]
    null_r_beta_lin = 10 ** NULL_R_BETA
        
    cog_mult = (live_r_beta_lin / null_r_beta_lin) if null_r_beta_lin > 0 else 1.0
    
    
    # Somatic Vagal Multiplier (Safe division fallback)
    som_mult = (NULL_HRV / session.state["rmssd"]) if session.state["rmssd"] > 0 else 1.0

    # 3. Composite AXX Score
    raw_axx = raw_ax * cog_mult * som_mult
    live_axx = raw_axx - AXX_OFFSET
    
    # Arousal (Beta) Smoothing (LOGARITHMIC FIX)
    avg_beta = (data["L_beta"] + data["R_beta"]) / 2
    avg_alpha = (data["L_alpha"] + data["R_alpha"]) / 2
    raw_ratio = avg_beta - avg_alpha # <--- FORENSIC LOG MATH
    session.state["ratio_raw"] = raw_ratio
    
    # Apply 0.95 smoothing (approx 1.5s visual lag)
    session.state["smooth_ratio"] = (session.state["smooth_ratio"] * 0.95) + (raw_ratio * 0.05)
    
    if session.is_calibrating:
        session.calib_buffer_faa.append(session.state["faa_raw"])
        if session.state["rmssd"] > 0: session.calib_buffer_hrv.append(session.state["rmssd"])
        if session.state["hr"] > 0: session.calib_buffer_hr.append(session.state["hr"])
        
        if len(session.calib_buffer_faa) > 1200: 
            if session.calib_buffer_hrv: BASE_HRV = sum(session.calib_buffer_hrv)/len(session.calib_buffer_hrv)
            if session.calib_buffer_hr: RESTING_HR = int(sum(session.calib_buffer_hr)/len(session.calib_buffer_hr))
            FAA_OFFSET = sum(session.calib_buffer_faa)/len(session.calib_buffer_faa)
            session.is_calibrating = False; is_hrv_locked = True 
            save_config(); session.hw.configure_baselines(RESTING_HR, BASE_HRV)
            session.outbound_messages.append({"type": "CALIB_DONE"})

    # 5. MATH (Uses smoothed inputs now)
    val, aro = normalize_metrics(session, session.state["faa_raw"], session.state["smooth_ratio"], session.state["rmssd"], session.state["hr"])
    session.x_buffer.append(val); session.y_buffer.append(aro)
    session.smooth_val = sum(session.x_buffer)/len(session.x_buffer)
    session.smooth_aro = sum(session.y_buffer)/len(session.y_buffer)
    session.zone_buffer_x.append(val); session.zone_buffer_y.append(aro)
    session.state["zone"] = determine_zone(sum(session.zone_buffer_x)/len(session.zone_buffer_x), sum(session.zone_buffer_y)/len(session.zone_buffer_y))
    
    # 6. HANDLE LABELS 
    current_label = session.state["annotation"]
    
    # Highest Priority: Artifact Suspension
    if session.state.get("phase_override") == "ARTIFACT":
        current_label = "ARTIFACT"
    elif session.state["stimulus_active"]:
        current_label = "STIMULUS" 
        session.state["stimulus_active"] = False 
    elif session.state["phase_label"] != "":
        current_label = session.state["phase_label"]
        
    # 7. HIGH-SPEED RECORDING (20Hz)
    if now - session.last_record_time > RECORD_INTERVAL:
        
        # --- 1-MINUTE AX TICKER ---
        session.ax_minute_buffer.append(live_ax)
        session.axx_minute_buffer.append(live_axx)
        
        if now - session.last_ax_tick >= 60.0:
            if len(session.ax_minute_buffer) > 0:
                avg_minute_ax = sum(session.ax_minute_buffer) / len(session.ax_minute_buffer)
                avg_minute_axx = sum(session.axx_minute_buffer) / len(session.axx_minute_buffer)
                
                ax_payload = {
                    "type": "AX_TICK", 
                    "val_ax": avg_minute_ax,
                    "val_axx": avg_minute_axx
                }
                session.outbound_messages.append(ax_payload)
                session.ax_minute_buffer.clear()
                session.axx_minute_buffer.clear()
            session.last_ax_tick = now

        if session.is_recording:
            # Capture Chaos Data
            if "CHAOS_" in session.state["phase_label"]:
                phase_key = session.state["phase_label"].replace("CHAOS_", "")
                if phase_key in session.chaos_phase_data and session.state["rmssd"] > 0:
                    session.chaos_phase_data[phase_key].append(session.state["rmssd"])

            session.total_samples += 1
            session.zone_counts[session.state["zone"]] += 1
            
            # --- NEW CSV HEADERS (Raw PPG Included) ---
            ppg = data.get("ppg_raw", [0, 0, 0])
            
            row = {
                "timestamp": datetime.now().isoformat(),
                "hr": session.state["hr"], "rmssd": float(session.state["rmssd"]),
                "sdnn": session.state["sdnn"],
                "valence": session.smooth_val, "arousal": session.smooth_aro, "zone": session.state["zone"],
                "raw_faa": session.state["faa_raw"],
                "raw_ax": raw_ax, "live_ax": live_ax,  
                "raw_axx": raw_axx, "live_axx": live_axx,
                "cog_mult": cog_mult, "som_mult": som_mult,
                "manual_label": current_label,
                "brain_weight": BRAIN_WEIGHT,
                "hr_source": hr_source,
                "base_hrv": BASE_HRV,
                "resting_hr": RESTING_HR,
                "L_delta": data["L_delta"], "R_delta": data["R_delta"],
                "L_theta": data["L_theta"], "R_theta": data["R_theta"],
                "L_alpha": data["L_alpha"], "R_alpha": data["R_alpha"],
                "L_beta": data["L_beta"],   "R_beta": data["R_beta"],
                "L_gamma": data["L_gamma"], "R_gamma": data["R_gamma"],
                "battery": data.get("battery", 0),
                "motion_gate": int(data.get("motion_gate", False)),
                "ppg_red": ppg[0] if len(ppg) > 0 else 0,
                "ppg_ir": ppg[1] if len(ppg) > 1 else 0,
                "ppg_green": ppg[2] if len(ppg) > 2 else 0
            }
            session.session_data.append(row)
        session.last_record_time = now

    # 8. LOW-SPEED BROADCAST (5Hz) -> Returns directly to JS caller
    if now - session.last_broadcast_time > BROADCAST_INTERVAL:
        
        avg_delta = (data["L_delta"] + data["R_delta"]) / 2
        avg_alpha = (data["L_alpha"] + data["R_alpha"]) / 2
        spec_ratio = avg_alpha / (avg_delta + 1e-6)
        
        if spec_ratio > 1.0: spec_status = "OPTIMAL" 
        elif spec_ratio > 0.5: spec_status = "OK"    
        else: spec_status = "NOISY"

        stats = {k: round((v/max(1,session.total_samples))*100, 1) for k,v in session.zone_counts.items()}
        
        payload = {
            "type": "LIVE", "x": session.smooth_val, "y": session.smooth_aro, "zone": session.state["zone"],
            "hr": session.state["hr"], "rmssd": float(session.state["rmssd"]),
            "raw_faa": session.state["faa_raw"], 
            "base_hrv": BASE_HRV, "resting_hr": RESTING_HR,
            "faa_offset": FAA_OFFSET, "ax_offset": AX_OFFSET, "axx_offset": AXX_OFFSET,
            "null_hrv": NULL_HRV, "null_r_beta": NULL_R_BETA,
            "brain_weight": BRAIN_WEIGHT, "is_hrv_locked": is_hrv_locked,
            "hr_source": hr_source,
            "L_beta": data.get('L_beta'),
            "R_beta": data.get('R_beta'),
            "spec_status": spec_status, 
            "stats": stats, "is_recording": session.is_recording, "status": "OK",
            "battery": data.get("battery", 0),
            "muse_batt": data.get("muse_batt", 0),
            "polar_batt": data.get("polar_batt", 0),
            "motion_gate": data.get("motion_gate", False),
        }
        
        # Attach any pending messages from the async protocols
        if session.outbound_messages:
            payload["alerts"] = session.outbound_messages.copy()
            session.outbound_messages.clear()
            
        session.last_broadcast_time = now
        return payload

    return None