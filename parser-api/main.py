from flask import Flask, request, jsonify
from google.cloud import storage
import uuid

app = Flask(__name__)

BUCKET_NAME = "music-omr-bucket-777135743132"

storage_client = storage.Client()

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_id = str(uuid.uuid4())
    pdf_path = f"input/{file_id}.pdf"

    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(pdf_path)
    blob.upload_from_file(file, content_type="application/pdf")

    return jsonify({
        "status": "uploaded",
        "pdf_gcs_uri": f"gs://{BUCKET_NAME}/{pdf_path}",
        "message": "Parser API is working."
    })

