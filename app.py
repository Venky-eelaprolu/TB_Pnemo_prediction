import streamlit as st
import numpy as np
import pandas as pd
import cv2
import pickle

# -------------------------
# Load trained model
# -------------------------
@st.cache_resource
def load_model():
    return pickle.load(open("TB_Pnemo_prediction_LOR.pkl", "rb"))

model = load_model()

# -------------------------
# Preprocess uploaded image
# (MUST match training!)
# -------------------------
def preprocess_uploaded(image):

    # Convert BGR → RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize same as training
    image = cv2.resize(image, (64, 64))

    # ⚠️ IMPORTANT: DO NOT NORMALIZE
    # Your model was trained on 0–255 pixel values
    img_array = np.array(image)

    # Flatten to 1 row
    img_flatten = img_array.flatten()

    # Convert to dataframe (1 sample)
    df = pd.DataFrame([img_flatten])

    return df


# -------------------------
# Streamlit UI
# -------------------------
st.title("Chest X-Ray Disease Classification")
st.write("Normal | Pneumonia | Tuberculosis")

uploaded = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    X = preprocess_uploaded(image)

    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    labels = model.classes_

    st.subheader(f"Prediction: **{pred}**")

    st.write("### Probabilities")
    for label, p in zip(labels, probs):
        st.write(f"- **{label}** : {p:.3f}")
