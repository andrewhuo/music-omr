import os
from flask import Flask, request, jsonify

app = Flask(__name__)

AUDIVERIS_HOME = os.environ.get("AUDIVERIS_HOME", "/usr/share/audiveris")

@app.route("/", methods=["GET"])
def health():
    return "omr-worker is running", 200

@app.route("/process", methods=["POST"])
def process():
    data = request.json or {}
    return jsonify({
        "status": "ok",
        "message": "Audiveris worker stub",
        "audiveris_home": AUDIVERIS_HOME,
        "received": data
    }), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
