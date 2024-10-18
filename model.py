import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Reshape
from tensorflow.keras.models import Model

def build_autoencoder(input_shape):
    input_img = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)
    encoded = Dense(1024, activation='relu')(x)
    x = Dense(128 * 8 * 8, activation='relu')(encoded)
    x = Reshape((8, 8, 128))(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder

def train_autoencoder(autoencoder, x_train, epochs=10, batch_size=32):
    autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, shuffle=True)

def save_model(autoencoder, filename='autoencoder_model.h5'):
    autoencoder.save(filename)

def load_model(filename='autoencoder_model.h5'):
    return tf.keras.models.load_model(filename)

if __name__ == '__main__':
    input_shape = (64, 64, 3)
    autoencoder = build_autoencoder(input_shape)
    x_train = np.random.rand(100, 64, 64, 3).astype(np.float32)  # Replace with actual data loading
    train_autoencoder(autoencoder, x_train, epochs=10, batch_size=32)
    save_model(autoencoder)
