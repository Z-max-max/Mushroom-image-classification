import os
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = BASE_DIR
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

MODEL_PATH = None
for fname in ["my_model.h5", "mushroom_model.h5", "model.h5"]:
    candidate = os.path.join(MODEL_DIR, fname)
    if os.path.exists(candidate):
        MODEL_PATH = candidate
        break

if MODEL_PATH is None:
    raise FileNotFoundError("ไม่พบไฟล์โมเดล (.h5)")

print(f"โหลดโมเดลจาก: {MODEL_PATH}")
model = load_model(MODEL_PATH)

CSV_PATH = None
for fname in ["mushrooms.csv", "labels.csv", "classes.csv"]:
    candidate = os.path.join(MODEL_DIR, fname)
    if os.path.exists(candidate):
        CSV_PATH = candidate
        break

df = None
map_by_numeric_label = {}

if CSV_PATH:
    print(f"โหลด CSV จาก: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    if {"class_id", "name", "description"}.issubset(df.columns):
        for _, row in df.iterrows():
            class_id = str(row["name"])  
            map_by_numeric_label[class_id] = {
                "name": str(row["class_id"]),  
                "description": str(row.get("description", "")),
            }

app = Flask(__name__, template_folder="templates", static_folder="static")


def prepare_image_for_model(path, target_size=None):
    if target_size is None:
        if hasattr(model, "input_shape"):
            target_size = (model.input_shape[1], model.input_shape[2])
        else:
            target_size = (224, 224)

    img = image.load_img(path, target_size=target_size)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = arr / 255.0
    return arr


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    if "file" not in request.files:
        return render_template("index.html", error="ไม่พบไฟล์ (field name ต้องเป็น 'file')")

    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", error="ไม่ได้เลือกไฟล์")

    filename = file.filename
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.seek(0)
    file.save(save_path)

    try:
        arr = prepare_image_for_model(save_path)
    except Exception as e:
        return render_template("index.html", error=f"เกิดข้อผิดพลาดขณะเตรียมรูป: {e}")

    try:
        preds = model.predict(arr)
    except Exception as e:
        return render_template("index.html", error=f"เกิดข้อผิดพลาดขณะทำนาย: {e}")

    pred_idx = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))

    key = str(pred_idx)
    info = map_by_numeric_label.get(
        key,
        {"name": f"class_{pred_idx}", "description": ""},
    )

    image_url = url_for("static", filename=f"uploads/{filename}")

    prediction = {
        "name": info.get("name", f"class_{pred_idx}"),       
        "class_id": str(pred_idx),                          
        "confidence": confidence,
        "description": info.get("description", "")
    }

    return render_template("index.html", prediction=prediction, image_file=image_url)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    return index()


if __name__ == "__main__":
    app.run(debug=True)
