import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import time
import uuid
import os
from datetime import datetime, timezone

app = Flask(__name__)
# Only the production site (and local dev servers on any port) may call this endpoint.
CORS(app, origins=["https://neurovis.io", re.compile(r"^http://(localhost|127\.0\.0\.1):\d+$")])

# SECURITY RULE 1: Max Payload Size (5 Megabytes)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

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

        # SECURITY RULE 2: Validate Required Keys
        # Matches the shape generateAnonymizedPayload() in NeurovisAW.html actually sends.
        if 'research_id' not in payload or 'daily_summaries' not in payload:
            return jsonify({"error": "Missing required data fields"}), 400

        if not isinstance(payload['daily_summaries'], list):
            return jsonify({"error": "'daily_summaries' must be a list"}), 400

        # SECURITY RULE 3: Date-Window Replay Sanity Check
        # upload_date is a YYYY-MM-DD string (day granularity), not a precise timestamp.
        upload_date_str = payload.get('upload_date')
        if not upload_date_str:
            return jsonify({"error": "Missing upload_date"}), 400

        try:
            upload_date = datetime.strptime(upload_date_str, "%Y-%m-%d").date()
        except ValueError:
            return jsonify({"error": "Invalid upload_date format"}), 400

        today = datetime.now(timezone.utc).date()
        if abs((today - upload_date).days) > 2:
            return jsonify({"error": "Payload expired. Possible replay attack."}), 403

        # SECURITY RULE 4: Safe Filename Generation
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