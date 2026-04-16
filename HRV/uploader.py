from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import hashlib
import time
import uuid
import os
from datetime import datetime, timezone

app = Flask(__name__)
# Allow cross-origin requests if your frontend is served on a different port
CORS(app) 

# SECURITY RULE 1: Max Payload Size (5 Megabytes)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 

# The secret salt shared with the frontend (keep this matching your JS!)
SECRET_SALT = "neurovis_alpha_2026"
SAVE_DIRECTORY = "./data_lake"

# Ensure our save directory exists
if not os.path.exists(SAVE_DIRECTORY):
    os.makedirs(SAVE_DIRECTORY)

@app.route('/api/upload', methods=['POST'])
def upload_data():
    try:
        # Check if payload is valid JSON
        if not request.is_json:
            return jsonify({"error": "Payload must be JSON"}), 400
        
        payload = request.get_json()
        headers = request.headers

        # SECURITY RULE 2: Validate Required Keys
        if 'hrv' not in payload or 'survey' not in payload:
            return jsonify({"error": "Missing required data arrays"}), 400

        # SECURITY RULE 3: Time-To-Live (TTL) Replay Prevention
        generated_at_str = payload.get('upload_generated_at')
        if not generated_at_str:
            return jsonify({"error": "Missing timestamp"}), 400
        
        generated_time = datetime.fromisoformat(generated_at_str.replace("Z", "+00:00"))
        current_time = datetime.now(timezone.utc)
        time_diff = (current_time - generated_time).total_seconds()
        
        if abs(time_diff) > 60: # 60 second expiration
            return jsonify({"error": "Payload expired. Possible replay attack."}), 403

        # SECURITY RULE 4: SHA-256 Payload Hash Verification
        client_hash = headers.get('X-Payload-Hash')
        if not client_hash:
            return jsonify({"error": "Missing payload signature"}), 403

        # Reconstruct the exact string the client hashed
        payload_string = json.dumps(payload, separators=(',', ':'))
        string_to_hash = payload_string + SECRET_SALT
        
        # Calculate our own hash and compare
        server_hash = hashlib.sha256(string_to_hash.encode('utf-8')).hexdigest()
        
        if client_hash != server_hash:
            return jsonify({"error": "Data tampering detected. Hashes do not match."}), 403

        # SECURITY RULE 5: Safe Filename Generation
        # Ignore whatever filename the user sent. Generate a random UUID.
        safe_filename = f"payload_{uuid.uuid4().hex[:8]}_{int(time.time())}.json"
        filepath = os.path.join(SAVE_DIRECTORY, safe_filename)

        with open(filepath, 'w') as f:
            json.dump(payload, f)

        return jsonify({"status": "success", "message": "Data securely vaulted in Data Lake."}), 200

    except Exception as e:
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Run the server on port 8002
    app.run(host='0.0.0.0', port=8002)