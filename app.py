import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.title("Malaria Cell Detection")
st.write("Cell ki image upload karen, model batayega ke wo infected hai ya nahi.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("malaria_cnn_model.keras")

model = load_model()

uploaded = st.file_uploader("Image upload karen", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", width=250)

    arr = np.array(img.resize((64, 64))) / 255.0
    arr = np.expand_dims(arr, axis=0)

    prob = float(model.predict(arr)[0][0])

    if prob > 0.5:
        st.success(f"Uninfected ({prob * 100:.1f}%)")
    else:
        st.error(f"Parasitized ({(1 - prob) * 100:.1f}%)")