import asyncio
from datetime import datetime
import math
import time
import numpy as np
from collections import deque

# --- NEW: WEB BRIDGE IMPORTS ---
import json
# -------------------------------

# --- NEW: LSL & DSP IMPORTS ---
from scipy.signal import welch, butter, filtfilt
# ------------------------------

# ==========================================
# CONFIGURATION
# ==========================================
OSC_IP = "0.0.0.0"
OSC_PORT = 5000
POLAR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
MUSE_TIMEOUT = 3.0
POLAR_TIMEOUT = 5.0

# MUSE FILTERS
MOTION_THRESHOLD = 2.5
MAX_BPM_JUMP = 10.0
MIN_INTER_BEAT = 0.30
RR_INPUT_SMOOTHING = 0.5
HRV_SMOOTHING_ALPHA = 0.2

class NeuroHardware:
    def __init__(self):
        self.state = {
            "muse_connected": False, "polar_connected": False,
            "system_status": "BOOTING", "quality_flag": "WAITING",
            "ts_muse": 0.0, "ts_polar": 0.0,
            "L_delta": 0, "L_theta": 0, "L_alpha": 0, "L_beta": 0, "L_gamma": 0,
            "R_delta": 0, "R_theta": 0, "R_alpha": 0, "R_beta": 0, "R_gamma": 0,
            "muse_batt": 0, "is_moving": False,
            "polar_hr": 0, "polar_rr_intervals": [], "polar_batt": 0,
            "muse_hr": 0, "muse_rr_intervals": [], "gyro_variance": 0.0, "rmssd": 0,
            # --- NEW DATA STREAMS ---
            "battery": 0, "motion_gate": False, "ppg_raw": [0, 0, 0],
            # --- ISSUE 2 FIX: Initialize empty raw states for LUNA ---
            "luna_raw_L": [], "luna_raw_R": []
        }
        self.polar_client = None
        self.accel_buffer = [] 
        
        # Signal State
        self.raw_history = [0.0] * 10
        self.baseline_hist = [0.0] * 50
        self.last_beat_time = 0
        self.peak_threshold = 0.0005
        self.was_rising = False
        self.prev_signal = 0.0
        
        # PRIMING STATE
        self.last_valid_bpm = 60 
        self.rmssd_smooth = 20    
        self.last_valid_timestamp = 0
        
        # Buffers
        self.muse_jitter_buffer = []
        self.muse_last_accepted_rr = 0
        self.muse_rr_hist = deque(maxlen=30)
        self.polar_rr_hist = deque(maxlen=30)
        self.gyro_history = []
        
        # Debug
        self.optics_packet_count = 0
        self.last_optics_packet = 0

        # --- NEW: DIRECT BLE RAW BUFFERS ---
        # Upgraded to deques to allow true 20Hz continuous processing
        self.lsl_buffer_L = deque(maxlen=256) # AF7 (Left Frontal)
        self.lsl_buffer_R = deque(maxlen=256) # AF8 (Right Frontal)
        # ----------------------------

        # --- ISSUE 2 FIX: Dedicated 3-Second Raw Waveform Buffers (256Hz * 3s = 768) ---
        self.luna_buffer_L = deque(maxlen=768)
        self.luna_buffer_R = deque(maxlen=768)
        # -------------------------------------------------------------------------------

    def configure_baselines(self, resting_hr, baseline_hrv):
        """
        Called by NeuroVis to Prime the Algorithm with User Stats.
        This prevents 'Cold Start' lag by giving the math a valid starting point.
        """
        print(f" HARDWARE: PRIMING WITH BASELINES -> HR:{resting_hr}, HRV:{baseline_hrv}")
        if resting_hr > 30: 
            self.last_valid_bpm = resting_hr
            self.state["muse_hr"] = resting_hr # Show baseline immediately on UI
        if baseline_hrv > 5:
            self.rmssd_smooth = baseline_hrv
            self.state["rmssd"] = baseline_hrv # Show baseline immediately on UI
    
    # --- NEW: FIT CHECK LOGIC ---
    async def perform_fit_check(self, duration=5):
        """
        Monitors signal for 'duration' seconds.
        Returns True if signal is balanced and stable. False if high noise/asymmetry.
        """
        print(f"   -> Waiting for Muse Stream (OSC)...")
        
        # 1. BLOCK UNTIL CONNECTION (No Timeout)
        # We check timestamps directly to avoid boolean race conditions
        while True:
            now = time.time()
            # If we received data in the last 2 seconds, we are live
            if (now - self.state["ts_muse"]) < 2.0 and self.state["ts_muse"] > 0:
                self.state["muse_connected"] = True
                print("   ->  MUSE SIGNAL DETECTED.")
                break
            await asyncio.sleep(1.0)
            
        print(f"   -> Sampling signal stability for {duration} seconds...")
        samples_L = []
        samples_R = []
        gyro_vals = []
        
        # 2. Collect samples
        for _ in range(duration * 10): # 10Hz sampling
            # Calculate Total Power (Delta+Theta+Alpha+Beta+Gamma)
            pL = sum([self.state[f"L_{b}"] for b in ["delta","theta","alpha","beta","gamma"]])
            pR = sum([self.state[f"R_{b}"] for b in ["delta","theta","alpha","beta","gamma"]])
            
            samples_L.append(pL)
            samples_R.append(pR)
            gyro_vals.append(self.state["gyro_variance"])
            await asyncio.sleep(0.1)
            
        # 3. Analysis
        avg_L = np.mean(samples_L)
        avg_R = np.mean(samples_R)
        avg_gyro = np.mean(gyro_vals)
        
        # Check 1: Is signal dead?
        if avg_L < 0.1 or avg_R < 0.1:
             print("   -> NO SIGNAL RECEIVED (Zeros detected). Check app.")
             return False
        
        # Check 2: Movement
        if avg_gyro > MOTION_THRESHOLD:
             print(f"   ->  TOO MUCH MOVEMENT ({avg_gyro:.2f}). Sit still.")
             return False
             
        # Check 3: Rail Check (Relaxed Fit)
        if avg_L > 1600 or avg_R > 1600:
            print(f"   ->  SENSOR RAILING. Adjust Headband.")
            return False
            
        return True

    async def _data_watchdog(self):
        print("🛡️ WATCHDOG: ACTIVE")
        while True:
            await asyncio.sleep(5)
            # Alert the user if the headband is streaming but the chest strap is missing
            if self.state["muse_connected"] and not self.state["polar_connected"]:
                print(f"\n[ ALERT] MUSE IS STREAMING, BUT NO HEART RATE DETECTED!")
                print("   -> ACTION: Please wet and connect your Polar H10 Chest Strap.\n")

    def get_latest(self):
        now = time.time()
        self.state["muse_connected"] = (now - self.state["ts_muse"]) < MUSE_TIMEOUT
        self.state["polar_connected"] = (now - self.state["ts_polar"]) < POLAR_TIMEOUT
        
        has_muse = self.state["muse_connected"]
        has_polar = self.state["polar_connected"]
        
        # --- NEW: MODULAR DEVICE GATEKEEPER ---
        if has_muse or has_polar:
            # If Muse is present, we still check it for motion artifacts
            if has_muse and (self.state["is_moving"] or self.state["gyro_variance"] > MOTION_THRESHOLD):
                self.state["system_status"] = "MOTION DETECTED"
                self.state["quality_flag"] = "NOISY"
            else:
                self.state["system_status"] = "SYSTEM ACTIVE"
                self.state["quality_flag"] = "OK"
        else:
            self.state["system_status"] = "WAITING FOR SENSORS"
            self.state["quality_flag"] = "LOST"
        # --------------------------------------

        # --- ISSUE 2 FIX: Safely pass the 3-second raw LUNA arrays to the state ---
        if len(self.luna_buffer_L) == 768:
            self.state["luna_raw_L"] = list(self.luna_buffer_L)
            self.state["luna_raw_R"] = list(self.luna_buffer_R)

        return self.state.copy()

    def _osc_band_handler(self, address, *args):
        if len(args) >= 4:
            mode = address.split("/")[-1].replace("_absolute", "")
            self.state[f"L_{mode}"] = args[2] 
            self.state[f"R_{mode}"] = args[3]
            self.state["ts_muse"] = time.time()

    def _osc_batt_handler(self, address, *args):
        if len(args) > 0: self.state["muse_batt"] = int(args[0] / 100) if args[0] > 100 else int(args[0])

    def _osc_accel_handler(self, address, *args):
        if len(args) >= 3:
            self.accel_buffer.append(np.sqrt(args[0]**2 + args[1]**2 + args[2]**2))
            if len(self.accel_buffer) > 10:
                self.state["is_moving"] = np.std(self.accel_buffer) > 0.15
                self.accel_buffer.pop(0)

    def _osc_gyro_handler(self, address, *args):
        if len(args) >= 3:
            mag = math.sqrt(args[0]**2 + args[1]**2 + args[2]**2)
            self.gyro_history.append(mag)
            if len(self.gyro_history) > 20: self.gyro_history.pop(0)
            if len(self.gyro_history) > 1:
                avg = sum(self.gyro_history) / len(self.gyro_history)
                self.state["gyro_variance"] = sum((x - avg) ** 2 for x in self.gyro_history) / len(self.gyro_history)

    def _osc_optics_handler(self, address, *args):
        if len(args) < 2: return
        self.last_optics_packet = time.time()
        
        if self.state["polar_connected"]: return

        is_stable = self.state["gyro_variance"] < MOTION_THRESHOLD
        raw_val = args[1] 
        
        # --- THE FIX: SYNTHESIZED HARDWARE TIMESTAMPS ---
        # We ignore network time and reconstruct perfect 64Hz timing
        if not hasattr(self, 'ppg_sample_count'):
            self.ppg_sample_count = 0
            self.last_beat_time = 0.0
            self.last_valid_timestamp = 0.0
            self.in_peak_region = False
            self.wave_peak_val = 0.0
            self.wave_peak_time = 0.0
            
        self.ppg_sample_count += 1
        timestamp = self.ppg_sample_count / 64.0 

        # 1. Smooth the raw signal (0.2s window @ 64Hz)
        self.raw_history.append(raw_val)
        if len(self.raw_history) > 13: self.raw_history.pop(0)
        smoothed_val = sum(self.raw_history) / len(self.raw_history)
        
        # 2. Dynamic Baseline (1.5s window @ 64Hz)
        self.baseline_hist.append(smoothed_val)
        if len(self.baseline_hist) > 96: self.baseline_hist.pop(0)
        
        if len(self.baseline_hist) < 30: return 
        moving_avg = sum(self.baseline_hist) / len(self.baseline_hist)

        # --- DEBUG LOGGING ---
        if self.ppg_sample_count % 64 == 0:  
            current_bpm = self.state.get("muse_hr", 0)
            print(f"🔎 [HEARTPY] Val: {smoothed_val:.0f} | MA: {moving_avg:.0f} | HW_BPM: {current_bpm} | SynthTime: {timestamp:.1f}s")

        # 3. Peak Region Detection
        is_above_ma = smoothed_val > moving_avg
        
        if is_above_ma:
            self.in_peak_region = True
            if smoothed_val > self.wave_peak_val:
                self.wave_peak_val = smoothed_val
                self.wave_peak_time = timestamp
        else:
            if self.in_peak_region:
                self.in_peak_region = False
                time_since_last_beat = self.wave_peak_time - self.last_beat_time
                
                # 4. Strict Refractory Lockout (0.4s = Max 150 BPM)
                if time_since_last_beat > 0.40: 
                    if is_stable:
                        raw_rr_ms = time_since_last_beat * 1000.0
                        bpm_est = 60000.0 / raw_rr_ms
                        
                        is_sane = 40 < bpm_est < 150
                        
                        self.muse_jitter_buffer.append(raw_rr_ms)
                        if len(self.muse_jitter_buffer) > 3: self.muse_jitter_buffer.pop(0)
                        
                        if is_sane and len(self.muse_jitter_buffer) == 3:
                            sorted_rr = sorted(self.muse_jitter_buffer)
                            median_rr = sorted_rr[1]
                            median_bpm = 60000.0 / median_rr
                            
                            self.muse_last_accepted_rr = median_rr
                            self.state["muse_hr"] = round(median_bpm)
                            
                            self.state["muse_rr_intervals"].append(median_rr)
                            
                            self.muse_rr_hist.append(median_rr)
                            if len(self.muse_rr_hist) >= 2:
                                diffs = np.diff(self.muse_rr_hist)
                                raw_hrv = np.sqrt(np.mean(diffs ** 2))
                                self.rmssd_smooth = (self.rmssd_smooth * (1.0 - HRV_SMOOTHING_ALPHA)) + (raw_hrv * HRV_SMOOTHING_ALPHA)
                                self.state["rmssd"] = int(self.rmssd_smooth)
                            
                            self.last_valid_timestamp = self.wave_peak_time 
                    
                    self.last_beat_time = self.wave_peak_time
                
                self.wave_peak_val = 0.0

    # ==========================================
    # --- NEW: DIRECT PAYLOAD INGESTION ---
    # ==========================================
    def process_web_payload(self, data):
        """ Receives parsed JSON from the central Neurovis.py router """
        if not hasattr(self, 'fit_status'):
            self.fit_status = {"AF7": "WAITING", "AF8": "WAITING"}
            self.last_fit_print = 0

        try:
            now = time.time()
            
            # 1. Update Global Connection Heartbeats
            if data.get("muse_connected"):
                self.state["muse_connected"] = True
                self.state["ts_muse"] = now
            if data.get("polar_connected"):
                self.state["polar_connected"] = True
                self.state["ts_polar"] = now

            # 2. Update Aux Sensors (Direct overwrite)
            if data.get("battery", 0) > 0:
                self.state["muse_batt"] = data.get("battery")
                self.state["battery"] = data.get("battery")
            if "polar_batt" in data and data.get("polar_batt") > 0:
                self.state["polar_batt"] = data.get("polar_batt")
                
            self.state["motion_gate"] = bool(data.get("gyro_mag", 0) > 50.0)

            # --- SCRUB GYRO ---
            for sample in data.get("gyro_raw", []):
                # Ensure it has 3 axes and none of them are None
                if len(sample) == 3 and all(s is not None for s in sample):
                    self._osc_gyro_handler("/web/gyro", sample[0], sample[1], sample[2])
                
            # Link the motion gate to your existing variance threshold
            self.state["motion_gate"] = self.state["gyro_variance"] > MOTION_THRESHOLD

            # Process Polar HR
            if "polar_hr" in data and data.get("polar_hr") is not None:
                self.state["polar_hr"] = data.get("polar_hr")

            # 3. Process Polar HRV
            for rr_ms in data.get("polar_rr", []):
                if rr_ms is not None and 300 < rr_ms < 1800:
                    self.state["polar_rr_intervals"].append(rr_ms)
                    self.polar_rr_hist.append(rr_ms)
                    if len(self.polar_rr_hist) >= 2:
                        diffs = np.diff(self.polar_rr_hist)
                        raw_hrv = np.sqrt(np.mean(diffs ** 2))
                        self.state["rmssd"] = float(raw_hrv)

            # --- SCRUB MUSE PPG ---
            for ir_val in data.get("muse_ppg", []):
                if ir_val is not None:
                    self._osc_optics_handler("/web/optics", 0, float(ir_val))

            # --- SCRUB HIGH-SPEED BRAINWAVES ---
            eeg_L = data.get("eeg_L", [])
            eeg_R = data.get("eeg_R", [])
            
            clean_L = []
            clean_R = []
            # Only keep frames where both left and right sensors reported data
            for l, r in zip(eeg_L, eeg_R):
                if l is not None and r is not None:
                    clean_L.append(l)
                    clean_R.append(r)
            
            # Use the cleaned data for the rest of the DSP math
            if clean_L and clean_R:
                self.lsl_buffer_L.extend(clean_L)
                self.lsl_buffer_R.extend(clean_R)
                
                # --- FIT CHECK LOGIC ---
                val_L, val_R = clean_L[-1], clean_R[-1]
                for name, val in [("AF7", val_L), ("AF8", val_R)]:
                    # AC-coupled signals center around 0, so absolute values are required
                    if abs(val) < 0.1: self.fit_status[name] = " DEAD (No Skin)"
                    elif abs(val) > 800.0: self.fit_status[name] = "⚠ RAILED (Hair/Static)"
                    else: self.fit_status[name] = " OK"
                    
                now_fit = time.time()
                if now_fit - self.last_fit_print > 2.0:
                    print(f" FIT CHECK | Left (AF7): {self.fit_status['AF7']} | Right (AF8): {self.fit_status['AF8']}")
                    self.last_fit_print = now_fit

                # --- EXACT LSL WELCH DSP MATCH ---
                if len(self.lsl_buffer_L) >= 256:
                    arr_L = np.array(self.lsl_buffer_L)
                    arr_R = np.array(self.lsl_buffer_R)
                    
                    # Apply 1-50Hz bandpass
                    b, a = butter(4, [1.0, 50.0], btype='bandpass', fs=256)
                    arr_L, arr_R = filtfilt(b, a, arr_L), filtfilt(b, a, arr_R)
                    
                    # Feed the filtered, noise-free frames to the LUNA buffer
                    filtered_new_L = arr_L[-len(clean_L):]
                    filtered_new_R = arr_R[-len(clean_R):]
                    self.luna_buffer_L.extend(filtered_new_L)
                    self.luna_buffer_R.extend(filtered_new_R)
                    
                    f, pxx_L = welch(arr_L, fs=256, nperseg=256, noverlap=128)
                    f, pxx_R = welch(arr_R, fs=256, nperseg=256, noverlap=128)
                    bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 50)}
                    
                    for band, (low, high) in bands.items():
                        idx = np.logical_and(f >= low, f <= high)
                        self.state[f"L_{band}"] = float(np.log10(max(1e-6, np.sum(pxx_L[idx]))))
                        self.state[f"R_{band}"] = float(np.log10(max(1e-6, np.sum(pxx_R[idx]))))
                        
        except Exception as e:
            print(f" PAYLOAD PROCESSING ERROR: {e}")