# # IMPORTS


import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from sklearn import datasets, metrics, model_selection, svm
import seaborn as sns
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
np.random.seed(31415)
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model


# # IMPORT DATASETS


signals = pd.read_csv(r'/home/scardoso/ResmMed4000mX1lb0p2yp0p4.csv')
BG = pd.read_csv(r'/home/scardoso/bkg.csv')

BG['LABEL'] = 0
signals['LABEL'] = 1


# # SPLITTING BG

# INPUTS
X_bg = BG.loc[:,['normalisedCombinedWeight','MET_px', 'MET_py','jet_e', 'jet_px', 'jet_py', 'jet_pz',
                  'ljet_e', 'ljet_px', 'ljet_py','ljet_pz','HT',
                  'gen_split','train_weight','LABEL']]  # REMOVER POR ENQUANTO jet_DL1r_max


# WEIGHTS OF bg_test
nCW_bg_test = (X_bg.loc[X_bg['gen_split'] == 'test'])['normalisedCombinedWeight']

# bg_train TO TRAIN AUTOENCODER
bg_train = (X_bg.loc[X_bg['gen_split'] == 'train']).drop(columns=['normalisedCombinedWeight',
                                                                  'gen_split','train_weight','LABEL'])

# bg_test TO JOIN TO signals_test TO TEST AUTOENCODER
bg_test = (X_bg.loc[X_bg['gen_split'] == 'test']).drop(columns=['normalisedCombinedWeight',
                                                                'gen_split','train_weight','LABEL'])

# bg_val TO JOIN TO signals_val TO VALIDATE AUTOENCODER
bg_val = (X_bg.loc[X_bg['gen_split'] == 'val']).drop(columns=['normalisedCombinedWeight',
                                                              'gen_split','train_weight','LABEL'])


# DEFINE LABELS TO USE FOR THE ROC CURVE
y_bg = BG.loc[:,['gen_split','LABEL']]

y_bg_train = (y_bg.loc[y_bg['gen_split'] == 'train']).drop(columns=['gen_split'])
y_bg_test = (y_bg.loc[y_bg['gen_split'] == 'test']).drop(columns=['gen_split'])
y_bg_val = (y_bg.loc[y_bg['gen_split'] == 'val']).drop(columns=['gen_split'])


# # SPLITTING SIGNALS

# INPUTS
X_signals = signals.loc[:,['normalisedCombinedWeight','MET_px', 'MET_py','jet_e', 'jet_px', 'jet_py', 'jet_pz',
                  'ljet_e', 'ljet_px', 'ljet_py','ljet_pz','HT',
                  'gen_split','train_weight','LABEL']]    # REMOVER POR ENQUANTO jet_DL1r_max



# WEIGHTS OF signals_test
nCW_signals_test = (X_signals.loc[X_signals['gen_split'] == 'test'])['normalisedCombinedWeight']



signals_train = (X_signals.loc[X_signals['gen_split'] == 'train']).drop(columns=['normalisedCombinedWeight',
                                                              'gen_split','train_weight','LABEL'])

# signals_test TO JOIN TO bg_test TO TEST AUTOENCODER
signals_test = (X_signals.loc[X_signals['gen_split'] == 'test']).drop(columns=['normalisedCombinedWeight',
                                                              'gen_split','train_weight','LABEL'])

# signals_val TO JOIN TO bg_val TO VALIDATE AUTOENCODER
signals_val = (X_signals.loc[X_signals['gen_split'] == 'val']).drop(columns=['normalisedCombinedWeight',
                                                              'gen_split','train_weight','LABEL'])


# DEFINE LABELS TO USE FOR THE ROC CURVE
y_signals = signals.loc[:,['gen_split','LABEL']]

y_signals_train = (y_signals.loc[y_signals['gen_split'] == 'train']).drop(columns=['gen_split'])
y_signals_test = (y_signals.loc[y_signals['gen_split'] == 'test']).drop(columns=['gen_split'])
y_signals_val = (y_signals.loc[y_signals['gen_split'] == 'val']).drop(columns=['gen_split'])


# # DEFINE WEIGHTS

# BG WEIGHTS
weight_bg_train = (X_bg.loc[X_bg['gen_split'] == 'train'])['train_weight'] 
weight_bg_test = (X_bg.loc[X_bg['gen_split'] == 'test'])['train_weight']
weight_bg_val = (X_bg.loc[X_bg['gen_split'] == 'val'])['train_weight']

# SIGNALS WEIGHTS
weight_signals_train = (X_signals.loc[X_signals['gen_split'] == 'train'])['train_weight']
weight_signals_test = (X_signals.loc[X_signals['gen_split'] == 'test'])['train_weight']
weight_signals_val = (X_signals.loc[X_signals['gen_split'] == 'val'])['train_weight']


# # SUM WEIGHTS

class_weights_train = (weight_bg_train.values.sum(),weight_signals_train.values.sum())
class_weights_test = (weight_bg_test.values.sum(),weight_signals_test.values.sum())
class_weights_val = (weight_bg_val.values.sum(),weight_signals_val.values.sum())

print("class_weights_train (BG, signals):",class_weights_train)
print("class_weights_test (BG, signals):",class_weights_test)
print("class_weights_val (BG, signals):",class_weights_val)


# # CONCATING DATASETS

# JOIN NECESSARY CSVs FOR X
X_train = bg_train
X_test = pd.concat([signals_test, bg_test], ignore_index=True)
X_val = bg_val

# JOIN NECESSARY CSVs FOR y
y_test = pd.concat([y_signals_test, y_bg_test], ignore_index=True).values
y_val = y_bg_val

# JOIN NECESSARY CSVs FOR weight
weight_train = weight_bg_train.values
weight_test = pd.concat([weight_signals_test, weight_bg_test], ignore_index=True).values.reshape(-1,1)
weight_val = weight_bg_val
nCW_test = pd.concat([nCW_signals_test, nCW_bg_test], ignore_index=True).values
nCW_test = nCW_test.reshape(-1,1)


# # STANDARDISATION OF INPUTS

from sklearn.preprocessing import StandardScaler

print("Original mean and variance:")
for feature, mean, std in zip(X_train.columns,X_train.mean(0), X_train.std(0)):
      print("{:9}: {:7.4f} +/- {:7.4f}".format(feature,mean,std))


scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train),columns = X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test),columns = X_test.columns)
X_val = pd.DataFrame(scaler.transform(X_val),columns = X_val.columns)


print("\nStandardised mean and variance:")
for feature, mean, std in zip(X_train.columns,X_train.mean(0), X_train.std(0)):
      print("{:9}: {:7.4f} +/- {:7.4f}".format(feature,mean,std))


# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------

# VAE - based on Keras website: 
# https://keras.io/examples/generative/vae/#display-how-the-latent-space-clusters-different-digit-classes
#
# Needs validation and callbacks implementation
# All variables standardised
# Currently running and learning

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
beta = 0.1                 # Constant that multiplies with KL_loss

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
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            # Mean Squared Error
            mse = tf.keras.losses.MeanSquaredError()
            reconstruction_loss = tf.reduce_mean(mse(data, reconstruction))   # tf.reduce_mean >> tf.reduce_sum ??
            
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


# Train the model

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(0.001))

#call = tf.keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)


history = vae.fit(X_train, epochs=30, batch_size=1024,) 


# history = vae.fit(X_train, epochs=30, batch_size=1024, validation_data=(X_val,weight_val), callbacks=[call])


# Plot losses
from matplotlib.pyplot import *

plt.figure()
plot(history.history['loss'],label="Training loss")
plot(history.history['reconstruction_loss'],label="Reconstruction loss")
plot(history.history['kl_loss'],label='KL loss')
plt.xlabel("Epoch")
plt.ylabel('Loss')
plt.legend()
plt.savefig('Loss')

