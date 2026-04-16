// ==========================================
// HARDWARE.JS (The Web Bluetooth Bridge)
// ==========================================

// Global State to match your old Python hardware dictionaries
let deviceState = {
    muse_connected: false,
    polar_connected: false,
    battery: 0,
    polar_hr: null
};

// High-speed data buffers (flushed to Pyodide @ 20Hz)
let buffers = {
    eeg_L: [],
    eeg_R: [],
    gyro_raw: [],
    muse_ppg: [],
    polar_rr: []
};

// Device References
let polarDevice = null;
let museDevice = null;

// ==========================================
// 1. POLAR H10 CONNECTION (Standard GATT Profile)
// ==========================================
async function connectPolar() {
    try {
        console.log("🔍 Scanning for Polar H10...");
        polarDevice = await navigator.bluetooth.requestDevice({
            filters: [{ namePrefix: 'Polar' }],
            optionalServices: ['heart_rate', 'battery_service']
        });

        polarDevice.addEventListener('gattserverdisconnected', () => {
            console.log("⚠️ Polar Disconnected");
            deviceState.polar_connected = false;
        });

        const server = await polarDevice.gatt.connect();
        deviceState.polar_connected = true;
        console.log("✅ Polar H10 Connected!");

        // 1. Get Heart Rate & RR Intervals
        const hrService = await server.getPrimaryService('heart_rate');
        const hrCharacteristic = await hrService.getCharacteristic('heart_rate_measurement');
        await hrCharacteristic.startNotifications();
        
        hrCharacteristic.addEventListener('characteristicvaluechanged', (event) => {
            const value = event.target.value;
            const flags = value.getUint8(0);
            const hr16Bit = flags & 0x01;
            
            // Extract HR
            let hr = hr16Bit ? value.getUint16(1, true) : value.getUint8(1);
            deviceState.polar_hr = hr;

            // Extract RR Intervals (if present in the BLE packet)
            const rrPresent = flags & 0x10;
            if (rrPresent) {
                let offset = hr16Bit ? 3 : 2;
                while (offset < value.byteLength) {
                    let rrRaw = value.getUint16(offset, true);
                    let rrMs = (rrRaw / 1024.0) * 1000.0;
                    buffers.polar_rr.push(rrMs);
                    offset += 2;
                }
            }
        });

        // 2. Get Battery
        try {
            const battService = await server.getPrimaryService('battery_service');
            const battChar = await battService.getCharacteristic('battery_level');
            const battValue = await battChar.readValue();
            deviceState.battery = battValue.getUint8(0);
        } catch (e) {
            console.log("Polar battery characteristic not readable.");
        }

    } catch (error) {
        console.error("Polar Connection Failed:", error);
    }
}

// ==========================================
// 2. MUSE CONNECTION (Using muse-js architecture)
// ==========================================
async function connectMuse() {
    try {
        console.log("🔍 Scanning for Muse...");
        // Muse uses a specific custom UUID for its serial data
        museDevice = await navigator.bluetooth.requestDevice({
            filters: [{ services: ['0fe11d01-14ce-4337-b434-5c8366de5b8c'] }]
        });

        museDevice.addEventListener('gattserverdisconnected', () => {
            console.log("⚠️ Muse Disconnected");
            deviceState.muse_connected = false;
        });

        const server = await museDevice.gatt.connect();
        deviceState.muse_connected = true;
        console.log("✅ Muse Connected!");

        // *************************************************************
        // NOTE: Decoding raw Muse BLE packets is highly complex (Base64/Protobuf). 
        // Most web-implementations use the 'muse-js' library here to map 
        // the GATT characteristics automatically.
        // 
        // When those packets arrive via muse-js observables, you just push 
        // them into our buffers array like this:
        //
        // buffers.eeg_L.push(af7_microvolts);
        // buffers.eeg_R.push(af8_microvolts);
        // buffers.gyro_raw.push([x, y, z]);
        // *************************************************************

    } catch (error) {
        console.error("Muse Connection Failed:", error);
    }
}

// ==========================================
// 3. THE PYODIDE BRIDGE (20Hz Engine Crank)
// ==========================================
let neurovisInterval = null;

function startNeurovisEngine(sessionId) {
    if (neurovisInterval) clearInterval(neurovisInterval);

    console.log(`🚀 Starting 20Hz Pyodide Bridge for Session: ${sessionId}`);

    neurovisInterval = setInterval(() => {
        // Ensure Pyodide is loaded before trying to crank the engine
        if (typeof pyodide === 'undefined' || !window.pyodideReady) return;

        // 1. Package the payload exactly how process_web_payload() expects it
        const payloadDict = {
            muse_connected: deviceState.muse_connected,
            polar_connected: deviceState.polar_connected,
            battery: deviceState.battery,
            polar_hr: deviceState.polar_hr,
            polar_rr: [...buffers.polar_rr],
            eeg_L: [...buffers.eeg_L],
            eeg_R: [...buffers.eeg_R],
            gyro_raw: [...buffers.gyro_raw],
            muse_ppg: [...buffers.muse_ppg]
        };

        // 2. Wipe the JS buffers so we don't send duplicate data to Python
        buffers.polar_rr = [];
        buffers.eeg_L = [];
        buffers.eeg_R = [];
        buffers.gyro_raw = [];
        buffers.muse_ppg = [];

        try {
            // 3. Translate JS Object -> Python Dictionary via Pyodide
            let pyPayload = pyodide.to_py(payloadDict);
            
            // 4. Call our new process_tick() function inside Python
            let pyResult = pyodide.globals.get('process_tick')(sessionId, pyPayload, null);
            
            // Destroy the temporary Python payload object to prevent browser memory leaks
            pyPayload.destroy(); 

            // 5. Unpack the Python Dictionary -> JS Object and update UI
            if (pyResult) {
                let jsResult = pyResult.toJs({ dict_converter: Object.fromEntries });
                pyResult.destroy();
                
                if (jsResult.live) {
                    updateDashboardUI(jsResult.live); // <--- Hooks into your frontend
                }
                
                if (jsResult.messages && jsResult.messages.length > 0) {
                    handlePythonAlerts(jsResult.messages); // <--- Hooks into your frontend prompts
                }
            }
        } catch (err) {
            console.error("Pyodide Engine Error:", err);
        }

    }, 50); // 50ms = 20Hz
}

// Stub functions so the file compiles - these will be defined in your main UI script
function updateDashboardUI(liveData) {
    // e.g., document.getElementById('hr-display').innerText = liveData.hr;
}

function handlePythonAlerts(messages) {
    // Handles the prompts from your async run_unified_session tests
}