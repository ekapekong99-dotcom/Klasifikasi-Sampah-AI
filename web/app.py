import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


app = Flask(__name__)


# ===============================
# PATH
# ===============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "model",
    "resnet50_best.keras"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# buat folder upload kalau belum ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ===============================
# LOAD MODEL
# ===============================

model = load_model(MODEL_PATH)


class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]


IMG_SIZE = (224, 224)


# ===============================
# HOME
# ===============================

@app.route("/")
def home():
    return render_template(
        "index.html"
    )


# ===============================
# PREDICT
# ===============================

@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return "Tidak ada file"

    file = request.files["file"]

    if file.filename == "":
        return "File kosong"

    # simpan gambar
    filename = file.filename

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(filepath)

    # preprocessing gambar
    img = image.load_img(
        filepath,
        target_size=IMG_SIZE
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = img_array / 255.0

    # prediksi
    prediction = model.predict(
        img_array
    )

    class_id = np.argmax(
        prediction
    )

    label = class_names[class_id]

    confidence = round(
        float(np.max(prediction))*100,
        2
    )

    return render_template(
        "index.html",
        prediction=label,
        confidence=confidence,
        image_path="uploads/" + filename
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
