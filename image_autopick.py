import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# =========================
# CONFIG
# =========================
MODEL_PATH = r"C:\Users\Aarav Gupta\OneDrive\Desktop\segregation and stuff\model_from_scratch.h5"
IMAGE_FOLDER = r"C:\Users\Aarav Gupta\OneDrive\Desktop\input_data"

IMG_SIZE = 224
CONF_THRESHOLD = 0.80
CLASSES = ["Recyclable", "Non-Recyclable"]

# =========================
# LOAD MODEL
# =========================
model = load_model(MODEL_PATH)
print("✅ Model loaded")

# =========================
# AUTO-PICK IMAGES
# =========================
if not os.path.exists(IMAGE_FOLDER):
    print("❌ Image folder does not exist")
    exit()

image_files = [
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if not image_files:
    print("❌ No images found in folder")
    exit()

print(f"📂 Found {len(image_files)} images")

# =========================
# PROCESS EACH IMAGE
# =========================
for file_name in image_files:

    image_path = os.path.join(IMAGE_FOLDER, file_name)
    print("\nProcessing:", file_name)

    # Windows-safe image loading
    img = cv2.imdecode(
        np.fromfile(image_path, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    if img is None:
        print("❌ Could not read image")
        continue

    # Preprocess
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    # Predict
    preds = model.predict(img_input, verbose=0)
    class_id = np.argmax(preds)
    confidence = float(np.max(preds))

    if confidence < CONF_THRESHOLD:
        result = "No waste detected"
    else:
        result = f"{CLASSES[class_id]} ({confidence*100:.1f}%)"

    print("Result:", result)

    # Show image
    cv2.putText(img, result, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Prediction", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
print("\n✅ Finished processing all images")
