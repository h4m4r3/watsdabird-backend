from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
from preprocessing.audio_util import AudioUtil

# ----------------------------
# Configuration: paths, allowed file types, model details, and audio parameters.
# ----------------------------
UPLOAD_FOLDER = "./data/uploads"
ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "ogg"}
MODEL_PATH = "./models/model.tflite"  # Ganti dari .keras ke .tflite
TARGET_SHAPE = (64, 108)
NUM_CLASSES = 4
SAMPLE_RATE = 22050
WINDOW_MS = 5000
OVERLAP_MS = 2500

# CLASS_MAP = {
#     0: "AtlanticCanary",
#     1: "Sooty-headedBulbul",
#     2: "ZebraDove",
#     3: "MoustachedBabbler",
# }

CLASS_MAP = {
    0: {
        "name": "AtlanticCanary",
        "common_name_id": "Kenari Atlantik",
        "scientific_name": "Serinus canaria",
        "genus": "Serinus",
        "subfamily": "Carduelinae",
        "family": "Fringillidae",
        "order": "Passeriformes",
        "description": "Kenari Atlantik adalah burung pengicau kecil yang berasal dari Kepulauan Canary, Madeira, dan Azores. Burung ini terkenal karena suaranya yang merdu dan sering dijadikan burung peliharaan.",
        "image_url": "https://www.whatbirdisthis.org/sites/default/files/14478544722_1f5b503f9f_z.jpg"
    },
    1: {
        "name": "Sooty-headedBulbul",
        "common_name_id": "Cucak Kutilang",
        "scientific_name": "Pycnonotus aurigaster",
        "genus": "Pycnonotus",
        "subfamily": "Pycnonotinae",
        "family": "Pycnonotidae",
        "order": "Passeriformes",
        "description": "Cucak kutilang adalah burung kicau populer di Indonesia, mudah dikenali dengan kepala hitam kelam dan suara kicau yang nyaring. Burung ini sering ditemukan di taman, pekarangan, serta area pepohonan atau semak belukar di sekitar pemukiman.",
        "image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRmAaM9zRiX_igmaAyffog-2ht-cKtesS-_X-evPShUSGYoODCgXzRQSHi7FWi4r05b3FRXlYGIkT7sR5P0dNAKDg"
    },
    2: {
        "name": "ZebraDove",
        "common_name_id": "Perkutut Zebra",
        "scientific_name": "Geopelia striata",
        "genus": "Geopelia",
        "subfamily": "Geopeliinae",
        "family": "Columbidae",
        "order": "Columbiformes",
        "description": "Perkutut Zebra adalah burung merpati kecil dengan pola garis-garis hitam putih di dada dan tubuh. Burung ini populer sebagai burung peliharaan di Asia Tenggara karena suaranya yang merdu dan menenangkan.",
        "image_url": "https://birdlifedata.blob.core.windows.net/species-images/22690708.jpg"
    },
    3: {
        "name": "MoustachedBabbler",
        "common_name_id": "Asi kumis",
        "scientific_name": "Malacopteron magnirostre",
        "genus": "Malacopteron",
        "subfamily": "Timaliinae",
        "family": "Pellorneidae",
        "order": "Passeriformes",
        "description": "Asi kumis adalah burung kecil yang hidup di kawasan tropis Asia Tenggara. Ciri khasnya adalah adanya 'kumis' putih di sisi wajah. Burung ini biasanya hidup dalam kelompok kecil dan suaranya sering terdengar di area berhutan lebat.",
        "image_url": "https://cdn.download.ams.birds.cornell.edu/api/v1/asset/462489281/1200"
    },
}

# ----------------------------
# Initialize Flask app and load the trained model.
# ----------------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("TFLite model loaded successfully.")
print(f"Input shape: {input_details[0]['shape']}")
print(f"Output shape: {output_details[0]['shape']}")


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


def predict_batch_tflite(batch: np.ndarray) -> np.ndarray:
    """
    Performs batch prediction using TFLite interpreter.
    
    Args:
        batch (np.ndarray): Batch of preprocessed spectrograms.
        
    Returns:
        np.ndarray: Prediction results for the batch.
    """
    batch_size = batch.shape[0]
    all_predictions = []
    
    # TFLite biasanya tidak mendukung batch processing secara langsung
    # Jadi kita perlu melakukan prediksi satu per satu
    for i in range(batch_size):
        # Prepare input data (add batch dimension jika diperlukan)
        input_data = batch[i:i+1].astype(np.float32)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        all_predictions.append(output_data[0])  # Remove batch dimension
    
    return np.array(all_predictions)


def predict_file(audio_path: str) -> tuple[list, int]:
    """
    Predicts the bird species in an audio file using a trained TFLite model.

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

    # Gunakan fungsi prediksi TFLite
    preds = predict_batch_tflite(batch)
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
                        CLASS_MAP.get(i, f"Class {i}")['name']: round(p, 4)
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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
