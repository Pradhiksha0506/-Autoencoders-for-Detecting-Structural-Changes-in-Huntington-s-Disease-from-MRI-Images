import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, smart_resize

# Load the autoencoder model
def load_model(filename='autoencoder_model.h5'):
    return tf.keras.models.load_model(filename)

# Preprocess the uploaded image
def preprocess_image(image, target_size):
    image = image.convert('RGB')  # Ensure the image is in RGB format
    image = img_to_array(image)  # Convert image to numpy array
    image = smart_resize(image, target_size)  # Resize image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image / 255.0  # Normalize the image

# Predict using the autoencoder
def predict(image, model):
    try:
        image = preprocess_image(image, (64, 64))  # Adjust size to match your model
        reconstruction = model.predict(image)
        # Simple logic for classification (you should customize this)
        is_affected = np.mean(reconstruction - image) > 0.1
        if is_affected:
            return "The model predicts that the person is unlikely to be affected by Huntington's Disease."
        else:
            return "The model predicts that the person might be affected by Huntington's Disease."
    except Exception as e:
        return f"Error during prediction: {str(e)}"

# Streamlit app
def main():
    st.set_page_config(page_title="Huntington's Disease Detection", page_icon="🧠", layout="centered")
    st.title("Huntington's Disease Detection")
    st.markdown('<style>body {background-color: #002b36;}</style>', unsafe_allow_html=True)  # Dark blue background

    st.write("Upload an MRI image to predict if the person might be affected by Huntington's Disease.")

    uploaded_file = st.file_uploader("Choose an MRI image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        
        model = load_model('autoencoder_model.h5')
        prediction = predict(image, model)
        st.write(prediction)

if __name__ == "__main__":
    main()
