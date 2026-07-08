import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# =====================================================
# PATH
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "model",
    "mobilenetv2_best.keras"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =====================================================
# LOAD MODEL
# =====================================================

model = load_model(MODEL_PATH)

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

IMG_SIZE = (224, 224)

# =====================================================
# HOME
# =====================================================


@app.route("/")
def home():
    return render_template("index.html")

# =====================================================
# PREDICT
# =====================================================


@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return render_template(
            "index.html",
            error="Silakan pilih gambar."
        )

    file = request.files["file"]

    if file.filename == "":
        return render_template(
            "index.html",
            error="Silakan pilih gambar."
        )

    filename = file.filename

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(filepath)

    img = image.load_img(
        filepath,
        target_size=IMG_SIZE
    )

    img = image.img_to_array(img)

    img = img / 255.0

    img = np.expand_dims(
        img,
        axis=0
    )

    prediction = model.predict(
        img,
        verbose=0
    )

    class_id = np.argmax(prediction)

    confidence = float(
        np.max(prediction)
    ) * 100

    label = CLASS_NAMES[class_id]

    return render_template(
        "index.html",
        prediction=label,
        confidence=round(confidence, 2),
        image_path="uploads/" + filename
    )

# =====================================================
# RUN
# =====================================================


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
