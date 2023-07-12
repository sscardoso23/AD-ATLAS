import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
np.random.seed(31415)
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
import os

# VAE - based on Keras website: 
# https://keras.io/examples/generative/vae/#display-how-the-latent-space-clusters-different-digit-classes

class Sampling(layers.Layer):
    # Uses (z_mean, z_log_var) to sample z, the vector encoding a digit

    def call(self, inputs):
        z_mean, z_log_var = inputs
        
        batch = tf.shape(z_mean)[0]   # tf.shape returns a 1-D integer tensor representing the shape of input
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))    # Outputs random values from a normal distribution.
        
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# Build the ENCODER

import keras

latent_dim = 2             # Number of latent space variables
shape = X_train.shape[1]   # Number of features
beta = 0.01                # Constant that multiplies with KL_loss

encoder_inputs = keras.Input(shape=(shape,))    # A shape tuple (integers), not including the batch size. 
                                                # For instance, shape=(11,) indicates that the expected 
                                                # input will be batches of 11-dimensional vectors.

x = layers.Dense(64, activation="relu")(encoder_inputs)
x = layers.Dense(32, activation="relu")(x)
x = layers.Dense(16, activation="relu")(x)

z_mean = layers.Dense(latent_dim, name="z_mean")(x)            # Latent space variables
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)      # Latent space variables

z = Sampling()([z_mean, z_log_var])                            # Latent space vector

encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()


# Build the DECODER

latent_inputs = keras.Input(shape=(latent_dim,))   # (None, 2)

x = layers.Dense(16, activation="relu")(latent_inputs)
x = layers.Dense(32, activation="relu")(x)
x = layers.Dense(64, activation="relu")(x)

decoder_outputs = layers.Dense(shape, activation=None)(x)   # Get back the initial number of features (None, 11)

decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()


# Custom train_step and define VAE as a Model

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.val_loss_tracker = keras.metrics.Mean(name='val_loss')


    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.val_loss_tracker,

        ]

    def train_step(self, data):
        X, sample_weight = data

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(X)
            reconstruction = self.decoder(z)
            
            # Mean Squared Error
            mse = tf.keras.losses.MeanSquaredError()
            reconstruction_loss = tf.reduce_mean(tf.math.multiply(mse(X, reconstruction), sample_weight)) # tf.reduce_mean better than tf.reduce_sum ??

            # KL Divergence Error
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(kl_loss)

            # Total Loss -> MSE + KL
            total_loss = reconstruction_loss + kl_loss*beta
            
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


    def test_step(self, input_data):
        validation_data, sample_weight = input_data

        z_mean, z_log_var, z = self.encoder(validation_data)
        val_reconstruction = self.decoder(z)

        # Mean Squared Error
        mse = tf.keras.losses.MeanSquaredError()
        val_reconstruction_loss = tf.reduce_mean(tf.math.multiply(mse(validation_data, val_reconstruction), sample_weight))

        # KL Divergence Error
        val_kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        val_kl_loss = tf.reduce_mean(val_kl_loss)

        # Total Loss -> MSE + KL
        val_total_loss = val_reconstruction_loss + val_kl_loss*beta

        self.val_loss_tracker.update_state(val_total_loss)

        return {"loss": self.val_loss_tracker.result()} 


    def custom_predict(self, data):
        _ , _ , z = np.array(self.encoder(data))
        preds = np.array(self.decoder(z))
        return preds

# Train the model

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(0.0001), weighted_metrics=[])

call = tf.keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)


history = vae.fit(X_train, weight_train, epochs=3, batch_size=1024, callbacks=[call], validation_data = (X_val, weight_val))

#----------
# LOSS PLOT
#----------

from matplotlib.pyplot import *

plt.figure()
plot(history.history['loss'],label="Training loss")
plot(history.history['reconstruction_loss'],label="Reconstruction loss")
plot(history.history['kl_loss'],label='KL loss')
plot(history.history['val_loss'], label='Validation loss')
plt.xlabel("Epoch")
plt.ylabel('Loss')
plt.legend()
plt.title('VAE Loss')
plt.savefig('/home/scardoso/data/Loss')
 
#-----------
# MODEL SAVE
#-----------
 
directory = "VAE"

parent_dir = "/home/scardoso/"

path = os.path.join(parent_dir, directory)
os.mkdir(path)

vae.encoder.save(f'/home/scardoso/VAE/encoder',save_format="tf")
vae.decoder.save(f'/home/scardoso/VAE/decoder',save_format="tf")
