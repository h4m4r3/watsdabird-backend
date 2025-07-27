from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
from keras_cv_attention_models import *
from tensorflow.keras.models import load_model
import tensorflow as tf
from preprocessing.audio_util import AudioUtil

# ----------------------------
# Configuration: paths, allowed file types, model details, and audio parameters.
# ----------------------------
UPLOAD_FOLDER = "./data/uploads"
ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "ogg"}
MODEL_PATH = "./models/model.keras"
TARGET_SHAPE = (64, 108)
NUM_CLASSES = 4
SAMPLE_RATE = 22050
WINDOW_MS = 5000
OVERLAP_MS = 2500

CLASS_MAP = {
    0: "AtlanticCanary",
    1: "Sooty-headedBulbul",
    2: "ZebraDove",
    3: "MoustachedBabbler",
}

# ----------------------------
# Initialize Flask app and load the trained model.
# ----------------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model(MODEL_PATH)
print("Model loaded successfully.")


def allowed_file(filename: str) -> bool:
    """
    Checks whether the file has a valid audio extension.

    Args:
        filename (str): The name of the uploaded file.

    Returns:
        bool: True if the file extension is allowed, False otherwise.
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_segment(segment_audio: tuple) -> np.ndarray:
    """
    Converts an audio segment into a normalized mel-spectrogram and resizes it
    for model compatibility.

    Args:
        segment_audio (tuple): Tuple of (audio waveform, sample rate).

    Returns:
        np.ndarray: Resized and normalized spectrogram with shape TARGET_SHAPE + (1,).
    """
    spec = AudioUtil.melspectrogram(segment_audio)
    spec = tf.image.resize(spec[..., np.newaxis], TARGET_SHAPE).numpy()
    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
    return spec


def predict_file(audio_path: str) -> tuple[list, int]:
    """
    Predicts the bird species in an audio file using a trained CNN model.

    The function:
    - Loads the audio file
    - Splits it into overlapping segments
    - Converts segments into spectrograms
    - Averages the model predictions over all segments

    Args:
        audio_path (str): Path to the input audio file.

    Returns:
        Tuple[list, int]: A tuple containing a list of class probabilities and
        the index of the predicted class.
    """
    audio = AudioUtil.open(audio_path, sample_rate=SAMPLE_RATE, mono=True)
    segments = AudioUtil.split(audio, WINDOW_MS, OVERLAP_MS)

    if not segments:
        raise ValueError("No valid audio segments found.")

    specs = [preprocess_segment(seg) for seg in segments]
    batch = np.stack(specs, axis=0)

    preds = model.predict(batch, verbose=0)
    avg_probs = preds.mean(axis=0)
    pred_idx = int(np.argmax(avg_probs))
    return avg_probs.tolist(), pred_idx


@app.route("/")
def hello() -> tuple[str, int]:
    """
    Root endpoint to verify that the server is running.

    Returns:
        Tuple[str, int]: A simple greeting message and HTTP status code 200.
    """
    return "Hello, World!", 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint. Accepts a POST request with an audio file,
    performs preprocessing and inference, and returns the predicted class
    along with the associated probabilities.

    Returns:
        JSON response containing prediction result or error message.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            probs, pred_idx = predict_file(filepath)
            os.remove(filepath)  # Clean up after prediction
            return jsonify(
                {
                    "prediction": CLASS_MAP.get(pred_idx, f"Class {pred_idx}"),
                    "probabilities": {
                        CLASS_MAP.get(i, f"Class {i}"): round(p, 4)
                        for i, p in enumerate(probs)
                    },
                }
            )
        except Exception as e:
            return jsonify({"error": f"Inference failed: {str(e)}"}), 500

    return jsonify({"error": "Unsupported file extension"}), 400


if __name__ == "__main__":
    """
    Runs the Flask server on localhost at port 5000.
    """
    app.run(host="0.0.0.0", port=5000, debug=True)
