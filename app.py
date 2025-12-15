
import streamlit as st
from PIL import Image
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import re
import requests



DEFAULT_MODEL_PATH = r"C:\Users\manav\OneDrive\Desktop\Project\acne_detection.keras"
INPUT_SIZE = (64, 64)  # fixed, do not expose to user

st.set_page_config(page_title="SkinCare AI — Face detector & recommender", layout="centered")


def load_keras_model(path=DEFAULT_MODEL_PATH):
    """Load a Keras model from disk. Returns model or None."""
    if tf is None:
        st.error("TensorFlow is not installed in this environment.")
        return None
    if not os.path.exists(path):
        return None
    try:
        model = load_model(path)
        return model
    except Exception as e:
        st.error(f"Failed to load Keras model from {path}: {e}")
        return None


def preprocess_image(img: Image.Image, target_size=INPUT_SIZE):
    """Resize to target_size, convert to RGB and scale to [0,1]. Returns shape (1,H,W,3)."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr



def predict_binary(model, img_arr, threshold=0.5, sigmoid_is_acne=True):
    
    if model is None:
        return None, None

    try:
        preds = model.predict(img_arr)
    except Exception:
        return None, None

    # Extract single scalar value
    try:
        raw = float(np.ravel(preds)[0])
    except Exception:
        return None, None

    # If model returns logits (outside [0,1]) apply sigmoid
    if raw < 0.0 or raw > 1.0:
        prob = 1.0 / (1.0 + np.exp(-raw))
    else:
        prob = raw

    # Interpret prob according to sigmoid_is_acne flag
    if sigmoid_is_acne:
        # prob = P(acne)
        if prob >= threshold:
            return "Normal", float(prob)
        else:
            return "Acne", float(1.0 - prob)
    else:
        # prob = P(normal)
        if prob >= threshold:
            return "Acne", float(prob)
        else:
            return "Normal", float(1.0 - prob)

    

    
def detect_faces(np_image):
    
    if cv2 is None:
        h, w = np_image.shape[:2]
        cw, ch = int(w * 0.6), int(h * 0.6)
        x = (w - cw) // 2
        y = (h - ch) // 2
        return [(x, y, cw, ch)]

    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            return []
        return faces.tolist()
    except Exception:
        # fallback central box
        h, w = np_image.shape[:2]
        cw, ch = int(w * 0.6), int(h * 0.6)
        x = (w - cw) // 2
        y = (h - ch) // 2
        return [(x, y, cw, ch)]


def draw_annotations(np_image, detections):
    
    img = np_image.copy()
    for d in detections:
        x, y, w, h = d['x'], d['y'], d['w'], d['h']
        label = d.get('label')
        conf = d.get('conf')
        if label is None:
            text = "Model error"
        else:
            text = f"{label}: {conf*100:.1f}%"

        # box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # text background
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        tx1, ty1 = x, max(0, y - th - 10)
        tx2, ty2 = x + tw + 6, y
        cv2.rectangle(img, (tx1, ty1), (tx2, ty2), (0, 255, 0), -1)

        # text
        cv2.putText(img, text, (x + 3, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def cv2_to_pil(bgr_img):
    """Convert BGR numpy image to PIL RGB image."""
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

# ---------------------- Product helpers ----------------------

def fetch_og_image(url):
    """Try to fetch og:image from a product page. Returns URL or None."""
    if requests is None:
        return None
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=6)
        html = r.text
        m = re.search(r'<meta property="og:image" content="([^"]+)"', html)
        if m:
            return m.group(1)
        m2 = re.search(r'"large":"(https:[^"]+)"', html)
        if m2:
            return m2.group(1).replace('\\u0026', '&')
        return None
    except Exception:
        return None

PRODUCT_DB = {
    'Acne': [
        { 'title': 'COSRX Salicylic Acid Daily Gentle Cleanser', 'url': 'https://www.amazon.in/Cosrx-Salicylic-Cleanser-milliliter-Skincare/dp/B0C5JS1LQM' },
        { 'title': 'Benzac AC 2.5% Gel (Benzoyl Peroxide)', 'url': 'https://www.amazon.in/s?k=benzoyl+peroxide+gel' },
        { 'title': 'La Roche-Posay Effaclar Duo+', 'url': 'https://www.amazon.in/s?k=La+Roche-Posay+Effaclar+Duo' }
    ],
    'Normal': [
        { 'title': 'Cetaphil Gentle Skin Cleanser', 'url': 'https://www.amazon.in/Cetaphil-Sulphate-Free-Hydrating-Niacinamide-Sensitive/dp/B01CCGW4OE' },
        { 'title': 'CeraVe Moisturizing Lotion', 'url': 'https://www.amazon.in/33599-CeraVe-Moisturising-Lotion-236ml/dp/B07CG2TD9F' },
        { 'title': 'La Roche-Posay Anthelios SPF50+', 'url': 'https://www.amazon.in/Roche-Posay-Anthelios-UVMune-Invisible-Resistant/dp/B09SLF5ZH8' }
    ]
}


def get_recs_with_images(key):
    out = []
    for p in PRODUCT_DB.get(key, []):
        img = fetch_og_image(p['url'])
        out.append({'title': p['title'], 'url': p['url'], 'img': img})
    return out

# ---------------------- Streamlit UI ----------------------

st.title("SkinCare Recommendation System")
st.markdown(
    "Upload or capture a photo. "
)

with st.sidebar:
    st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background: linear-gradient(#87CEEB, #B0E0E6);
            width: 145px !important;
            
        }
    </style>
""", unsafe_allow_html=True)

    st.sidebar.markdown(r"""
    <div style='text-align: left;'>
        <h1 style = "color:black;">Welcome!! </h1>
        <h4 style = "color:black;">Listen to your SKINN...</h4>
    </div>""",unsafe_allow_html=True)


# auto-load model into session state once
if 'model' not in st.session_state:
    st.session_state.model = None
    if os.path.exists(DEFAULT_MODEL_PATH) and tf is not None:
        st.session_state.model = load_keras_model(DEFAULT_MODEL_PATH)
        if st.session_state.model is not None:
            st.success(f"Loaded model from {DEFAULT_MODEL_PATH}")
        else:
            st.warning(f"Found file at {DEFAULT_MODEL_PATH} but failed to load as Keras model.")
    else:
        if not os.path.exists(DEFAULT_MODEL_PATH):
            st.info(f"Model not found at {DEFAULT_MODEL_PATH}. Place your .keras file there.")

col1, col2 = st.columns([2, 1])
with col1:
    camera_img = st.camera_input("Take a selfie")
    uploaded_file = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])

with col2:
    st.info("Tips: face should be well-lit and not heavily filtered.")

image = None
if camera_img is not None:
    image = Image.open(camera_img)
elif uploaded_file is not None:
    image = Image.open(uploaded_file)

if image is not None:
    st.image(image, caption="Input image", use_container_width=True)

    if st.button("Analyse"):
        with st.spinner("Running detection and classification..."):
            model = st.session_state.model
            if model is None:
                st.error("Model not loaded. Please place /mnt/data/acne_detection.keras and reload the app.")
                st.stop()

            np_img = np.array(image.convert('RGB'))[:, :, ::-1].copy()  # RGB->BGR
            faces = detect_faces(np_img)
            detections = []
            any_acne = False

            for (x, y, w, h) in faces:
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(x + w, np_img.shape[1])
                y2 = min(y + h, np_img.shape[0])
                if x2 <= x1 or y2 <= y1:
                    continue

                face_crop = np_img[y1:y2, x1:x2]
                face_pil = Image.fromarray(face_crop[:, :, ::-1])
                arr = preprocess_image(face_pil)
                label, conf = predict_binary(model, arr)
                detections.append({'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1, 'label': label, 'conf': conf})
                if label == 'Acne':
                    any_acne = True

            if len(detections) == 0:
                st.warning("No faces detected.")

            annotated = draw_annotations(np_img, detections)
            annotated_pil = cv2_to_pil(annotated) if cv2 is not None else Image.fromarray(np_img[:, :, ::-1])

        st.subheader("Detected faces & predictions")
        st.image(annotated_pil, use_container_width=True)

        st.write("---")
        for i, d in enumerate(detections, start=1):
            lbl = d['label'] if d['label'] is not None else 'Model error'
            conf_text = f" — confidence: {d['conf']*100:.1f}%" if d['conf'] is not None else ''
            st.write(f"Face {i}: {lbl}{conf_text}")

        # ---------------------- Skin care tips ----------------------
        st.write("---")
        st.header("Skin care tips based on result")
        if any_acne:
            st.markdown(
                """
                ### Tips for acne-prone skin
                - Use oil-free, non-comedogenic products.
                - Cleanse twice daily with a gentle salicylic acid cleanser.
                - Avoid picking or popping pimples.
                - Consider benzoyl peroxide spot treatment at night.
                - Change pillow covers every 2–3 days.
                - Apply broad-spectrum SPF 30+ sunscreen every morning.
                - If severe or persistent, consult a dermatologist.
                """
            )
        else:
            st.markdown(
                """
                ### Tips for normal/healthy skin
                - Keep a simple routine: cleanser → moisturizer → sunscreen.
                - Do not overwash; twice a day is sufficient.
                - Use a lightweight, hydrating moisturizer.
                - Apply sunscreen daily (SPF 30+).
                - Avoid harsh scrubs unless needed.
                - Maintain a balanced diet and hydration.
                """
            )

        # ---------------------- Product recommendations ----------------------
        st.write("---")
        st.header("Product recommendations")
        label_key = 'Acne' if any_acne else 'Normal'
        recs = get_recs_with_images(label_key)
        st.write("Click product title or image to open the Amazon page.")

        for prod in recs:
            cols = st.columns([1, 3])
            if prod['img']:
                try:
                    cols[0].image(prod['img'], width=120)
                except Exception:
                    cols[0].write("Image unavailable")
            else:
                cols[0].write("No image")

            cols[1].markdown(f"**[{prod['title']}]({prod['url']})**")
            cols[1].markdown(f"[Buy on Amazon]({prod['url']})")

        if st.button("Save annotated image"):
            save_path = "annotated_result.jpg"
            annotated_pil.convert('RGB').save(save_path)
            st.success(f"Saved annotated image to {save_path}")

else:
    st.warning("No image — upload or take a selfie.")
st.caption("Disclaimer: for experimentation only. Not medical advice.")