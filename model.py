import streamlit as st
import numpy as np
import cv2
import os
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.preprocessing.image import img_to_array, smart_resize, ImageDataGenerator
from sklearn.model_selection import train_test_split


@st.cache_resource
def train_and_save_model():
    affected_img_path = Path("D:/final year/aff.jpg")
    unaffected_img_path = Path("D:/final year/nor.jpeg")

    affected_img = cv2.imread(str(affected_img_path))
    unaffected_img = cv2.imread(str(unaffected_img_path))

    if affected_img is None or unaffected_img is None:
        raise FileNotFoundError("Check the image paths. One or more images not loaded properly.")

    affected_img = cv2.resize(affected_img, (256, 256)).astype("float32") / 255.0
    unaffected_img = cv2.resize(unaffected_img, (256, 256)).astype("float32") / 255.0

    images = np.array([affected_img, unaffected_img])

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    augmented_images = []
    for img in images:
        img = np.expand_dims(img, axis=0)
        it = datagen.flow(img, batch_size=1)
        for _ in range(100):  
            batch = next(it)
            augmented_images.append(batch[0])

    augmented_images = np.array(augmented_images)
    train_images, test_images = train_test_split(augmented_images, test_size=0.3, random_state=42)
    val_images, test_images = train_test_split(test_images, test_size=0.5, random_state=42)

    def build_autoencoder(input_shape=(256, 256, 3)):
        input_img = Input(shape=input_shape)
        x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
        x = MaxPooling2D((2, 2), padding="same")(x)
        x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = MaxPooling2D((2, 2), padding="same")(x)
        x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        encoded = MaxPooling2D((2, 2), padding="same")(x)

        x = Conv2D(128, (3, 3), activation="relu", padding="same")(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer="adam", loss="mse")
        return autoencoder

    autoencoder = build_autoencoder()
    autoencoder.fit(
        train_images, train_images,
        epochs=5,
        batch_size=16,
        validation_data=(val_images, val_images)
    )

    autoencoder.save("hd_detection_autoencoder_safe.h5")
    np.save("train_images.npy", train_images)
    return autoencoder, train_images


def preprocess_image(image, target_size=(256, 256)):
    image = image.convert('RGB')
    image = img_to_array(image)
    image = smart_resize(image, target_size)
    image = np.expand_dims(image, axis=0)
    return image / 255.0


def compute_threshold(model, train_images):
    reconstructions = model.predict(train_images)
    reconstruction_errors = np.mean(np.abs(reconstructions - train_images), axis=(1, 2, 3))
    threshold = np.percentile(reconstruction_errors, 95)
    return threshold


def predict(image, model, threshold):
    try:
        image = preprocess_image(image)
        reconstruction = model.predict(image)
        difference = np.mean(np.abs(reconstruction - image))

        is_affected = difference > threshold

        if is_affected:
            status = "🔴 **Likely Affected**"
            explanation = (
                "The MRI image shows structural deviations from the norm that may correlate with "
                "Huntington's Disease. Please consult a medical professional for further diagnosis."
            )
        else:
            status = "🟢 **Likely Unaffected**"
            explanation = (
                "The MRI image appears structurally normal. No significant anomalies were detected by the model. "
                "However, this is not a clinical diagnosis—consult a neurologist for confirmation."
            )

        return (
            f"### Prediction Result: {status}\n"
            f"- **Anomaly Score**: `{difference:.4f}`\n"
            f"- **Threshold Used**: `{threshold:.4f}`\n\n"
            f"### Interpretation:\n{explanation}"
        )

    except Exception as e:
        return f"Prediction error: {e}"




def main():
    st.set_page_config(page_title="HD Detection", page_icon="🧠")
    st.title("🧠 Huntington's Disease Detection from MRI")
    st.markdown("Upload an MRI scan to check for possible structural changes related to Huntington's Disease.")

    uploaded_file = st.file_uploader("Upload an MRI image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Image", use_column_width=True)

        st.write("⚙️ Preparing model...")

        
        if Path("hd_detection_autoencoder_safe.h5").exists():
            model = load_model("hd_detection_autoencoder_safe.h5", compile=False)
            if Path("train_images.npy").exists():
                train_images = np.load("train_images.npy")
            else:
                model, train_images = train_and_save_model()
        else:
            model, train_images = train_and_save_model()

        
        st.markdown("### 🔧 Threshold Configuration")
        use_dynamic = st.checkbox("Use dynamic threshold (based on training data)", value=True)

        if use_dynamic:
            threshold = compute_threshold(model, train_images)
        else:
            threshold = 0.02  

        st.write(f"📊 Threshold set to: `{threshold:.4f}`")

        
        with st.spinner("🔍 Analyzing image..."):
            prediction = predict(image, model, threshold)

        st.success(prediction)


if __name__ == "__main__":
    main()
