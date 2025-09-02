import streamlit as st
import numpy as np
import time
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image

# ✅ Page config must be the first Streamlit command
st.set_page_config(page_title="Image Classifier", page_icon="🖼️", layout="centered")

# Load pre-trained ResNet50 model
@st.cache_resource
def load_model():
    return ResNet50(weights='imagenet')

model = load_model()

# Streamlit UI
st.title("🖼️ ResNet50 Image Classifier")
st.write("Upload an image and the model will classify it using **ImageNet classes**.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Inference
    start_time = time.time()
    predictions = model.predict(img_array)
    end_time = time.time()

    # Decode predictions
    decoded_predictions = decode_predictions(predictions, top=5)[0]

    st.subheader("🔮 Predictions")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        st.write(f"**{i+1}. {label}** - {score:.2f}")

    # Extra info
    st.write("---")
    st.write(f"⏱ Inference Time: {(end_time - start_time) * 1000:.2f} ms")
    st.write(f"📦 Model Size: {model.count_params() * 4 / (1024**2):.2f} MB")
    st.write(f"🧩 Parameters: {model.count_params():,}")
    st.write(f"🔗 Depth (layers): {len(model.layers)}")
