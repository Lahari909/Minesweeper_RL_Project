import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam

def create_dqn(learn_rate, in_dims, n_actions, conv_units, dense_units):
    model = Sequential([
        Conv2D(filters=conv_units, kernel_size=(3, 3), activation='relu', padding='same', input_shape=in_dims),
        Conv2D(filters=conv_units, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(filters=conv_units, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(filters=conv_units, kernel_size=(3, 3), activation='relu', padding='same'),
        Flatten(),
        Dense(dense_units, activation='relu'),
        Dense(dense_units, activation='relu'),
        Dense(n_actions, activation='linear')
    ])
    
    model.compile(optimizer=Adam(learning_rate=learn_rate, epsilon=1e-4), loss='mse')
    return model
