import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from sklearn import datasets, metrics, model_selection, svm
import seaborn as sns
import matplotlib.pyplot as plt
np.random.seed(31415)
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import os
import keras


#----------------
# IMPORT DATASETS
#----------------


BG = pd.read_csv('/home/scardoso/bkg.csv')
s1 = pd.read_csv('/home/scardoso/ResmMed4000mX1lb0p2yp0p4.csv')
s2 = pd.read_csv('/home/scardoso/bbA2000_yb2_Zhvvbb.csv')
s3 = pd.read_csv('/home/scardoso/GG_direct_2000_0.csv')
s4 = pd.read_csv('/home/scardoso/HVT_Agv1_VzZH_vvqq_m1000.csv')

BG['LABEL'] = 0
s1['LABEL'] = 1
s2['LABEL'] = 1
s3['LABEL'] = 1
s4['LABEL'] = 1


#----------------
# CONCAT DATASETS
#----------------


data_1 = pd.concat([s1,BG], ignore_index=True)
data_2 = pd.concat([s2,BG], ignore_index=True)
data_3 = pd.concat([s3,BG], ignore_index=True)
data_4 = pd.concat([s4,BG], ignore_index=True)


#---------------
# SPLIT DATASETS
#---------------


X_1 = data_1.loc[:,['normalisedCombinedWeight','MET', 'MET_Phi','jet_pt', 
                 'jet_e', 'jet_eta', 'jet_phi', 'ljet_pt', 'topjet_pt',
                 'ljet_e', 'topjet_e', 'ljet_m', 'topjet_m', 'ljet_eta', 'topjet_eta',
                 'ljet_phi', 'topjet_phi', 'Omega', 'HT', 'Centrality',
                 'DeltaR_max', 'jet_DL1r_max', 'jet_px', 'jet_py',
                 'jet_pz', 'ljet_px', 'ljet_py', 'ljet_pz', 'MET_m', 'MET_eta', 'MET_px',
                 'MET_py','gen_split','train_weight','LABEL']]

X_train_1 = (X_1.loc[X_1['gen_split']=='train']).drop(columns=['normalisedCombinedWeight','gen_split','train_weight','LABEL'])
X_val_1 = (X_1.loc[X_1['gen_split']=='val']).drop(columns=['normalisedCombinedWeight','gen_split','train_weight','LABEL'])

nCW_train_1 = (X_1.loc[X_1['gen_split']=='train'])['normalisedCombinedWeight']
nCW_val_1 = (X_1.loc[X_1['gen_split']=='val'])['normalisedCombinedWeight']

#

X_2 = data_2.loc[:,['normalisedCombinedWeight','MET', 'MET_Phi','jet_pt', 
                 'jet_e', 'jet_eta', 'jet_phi', 'ljet_pt', 'topjet_pt',
                 'ljet_e', 'topjet_e', 'ljet_m', 'topjet_m', 'ljet_eta', 'topjet_eta',
                 'ljet_phi', 'topjet_phi', 'Omega', 'HT', 'Centrality',
                 'DeltaR_max', 'jet_DL1r_max', 'jet_px', 'jet_py',
                 'jet_pz', 'ljet_px', 'ljet_py', 'ljet_pz', 'MET_m', 'MET_eta', 'MET_px',
                 'MET_py','gen_split','train_weight','LABEL']]

X_train_2 = (X_2.loc[X_2['gen_split']=='train']).drop(columns=['normalisedCombinedWeight','gen_split','train_weight','LABEL'])
X_val_2 = (X_2.loc[X_2['gen_split']=='val']).drop(columns=['normalisedCombinedWeight','gen_split','train_weight','LABEL'])

nCW_train_2 = (X_2.loc[X_2['gen_split']=='train'])['normalisedCombinedWeight']
nCW_val_2 = (X_2.loc[X_2['gen_split']=='val'])['normalisedCombinedWeight']

#

X_3 = data_3.loc[:,['normalisedCombinedWeight','MET', 'MET_Phi','jet_pt', 
                 'jet_e', 'jet_eta', 'jet_phi', 'ljet_pt', 'topjet_pt',
                 'ljet_e', 'topjet_e', 'ljet_m', 'topjet_m', 'ljet_eta', 'topjet_eta',
                 'ljet_phi', 'topjet_phi', 'Omega', 'HT', 'Centrality',
                 'DeltaR_max', 'jet_DL1r_max', 'jet_px', 'jet_py',
                 'jet_pz', 'ljet_px', 'ljet_py', 'ljet_pz', 'MET_m', 'MET_eta', 'MET_px',
                 'MET_py','gen_split','train_weight','LABEL']]

X_train_3 = (X_3.loc[X_3['gen_split']=='train']).drop(columns=['normalisedCombinedWeight','gen_split','train_weight','LABEL'])
X_val_3 = (X_3.loc[X_3['gen_split']=='val']).drop(columns=['normalisedCombinedWeight','gen_split','train_weight','LABEL'])

nCW_train_3 = (X_3.loc[X_3['gen_split']=='train'])['normalisedCombinedWeight']
nCW_val_3 = (X_3.loc[X_3['gen_split']=='val'])['normalisedCombinedWeight']

#

X_4 = data_4.loc[:,['normalisedCombinedWeight','MET', 'MET_Phi','jet_pt', 
                 'jet_e', 'jet_eta', 'jet_phi', 'ljet_pt', 'topjet_pt',
                 'ljet_e', 'topjet_e', 'ljet_m', 'topjet_m', 'ljet_eta', 'topjet_eta',
                 'ljet_phi', 'topjet_phi', 'Omega', 'HT', 'Centrality',
                 'DeltaR_max', 'jet_DL1r_max', 'jet_px', 'jet_py',
                 'jet_pz', 'ljet_px', 'ljet_py', 'ljet_pz', 'MET_m', 'MET_eta', 'MET_px',
                 'MET_py','gen_split','train_weight','LABEL']]

X_train_4 = (X_4.loc[X_4['gen_split']=='train']).drop(columns=['normalisedCombinedWeight','gen_split','train_weight','LABEL'])
X_val_4 = (X_4.loc[X_4['gen_split']=='val']).drop(columns=['normalisedCombinedWeight','gen_split','train_weight','LABEL'])

nCW_train_4 = (X_4.loc[X_4['gen_split']=='train'])['normalisedCombinedWeight']
nCW_val_4 = (X_4.loc[X_4['gen_split']=='val'])['normalisedCombinedWeight']


#----------------
# STANDARD SCALER
#----------------


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_1 = pd.DataFrame(scaler.fit_transform(X_train_1),columns = X_train_1.columns)
X_val_1 = pd.DataFrame(scaler.transform(X_val_1),columns = X_val_1.columns)

X_train_2 = pd.DataFrame(scaler.fit_transform(X_train_2),columns = X_train_2.columns)
X_val_2 = pd.DataFrame(scaler.transform(X_val_2),columns = X_val_2.columns)

X_train_3 = pd.DataFrame(scaler.fit_transform(X_train_3),columns = X_train_3.columns)
X_val_3 = pd.DataFrame(scaler.transform(X_val_3),columns = X_val_3.columns)

X_train_4 = pd.DataFrame(scaler.fit_transform(X_train_4),columns = X_train_4.columns)
X_val_4 = pd.DataFrame(scaler.transform(X_val_4),columns = X_val_4.columns)


#------------------------
# VARIATIONAL AUTOENCODER
#------------------------


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

latent_dim = 11              # Number of latent space variables
shape = X_train_1.shape[1]   # Number of features (31)
beta = 0.1                   # Constant that multiplies with KL_loss

encoder_inputs = keras.Input(shape=(shape,))    # A shape tuple (integers), not including the batch size. 
                                                # For instance, shape=(31,) indicates that the expected 
                                                # input will be batches of 31-dimensional vectors.

x = layers.Dense(32, activation="relu")(encoder_inputs)
x = layers.Dropout(0.2, name="Droput_0")(x)
x = layers.Dense(16, activation="relu")(x)
x = layers.Dropout(0.2, name="Droput_1")(x)

z_mean = layers.Dense(latent_dim, name="z_mean")(x)            # Latent space variables (None,8)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)      # Latent space variables (None,8)

z = Sampling()([z_mean, z_log_var])                            # Latent space vector (None,8)

encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

# Build the DECODER

latent_inputs = keras.Input(shape=(latent_dim,))   # (None, 8)


x = layers.Dense(16, activation="relu")(latent_inputs)
x = layers.Dropout(0.2, name="Droput_2")(x)
x = layers.Dense(32, activation="relu")(x)
x = layers.Dropout(0.2, name="Droput_3")(x)

decoder_outputs = layers.Dense(shape, activation=None)(x)   # Get back the initial number of features (None, 31)

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
            self.val_loss_tracker,]

    def train_step(self, data):
        X, sample_weight = data

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(X)
            reconstruction = self.decoder(z)
            
            # Mean Squared Error
            func = tf.keras.losses.MeanSquaredError()
            reconstruction_loss = 1e3*tf.reduce_mean(func(X, reconstruction, sample_weight))

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
        func = tf.keras.losses.MeanSquaredError()
        val_reconstruction_loss = 1e3*tf.reduce_mean(func(validation_data, val_reconstruction, sample_weight))

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


#---------------
# FITTING MODELS
#---------------


vae_1 = VAE(encoder, decoder)
vae_2 = VAE(encoder, decoder)
vae_3 = VAE(encoder, decoder)
vae_4 = VAE(encoder, decoder)

vae_1.compile(optimizer=keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95, epsilon=1e-08, decay=0.0), weighted_metrics=[])
vae_2.compile(optimizer=keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95, epsilon=1e-08, decay=0.0), weighted_metrics=[])
vae_3.compile(optimizer=keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95, epsilon=1e-08, decay=0.0), weighted_metrics=[])
vae_4.compile(optimizer=keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95, epsilon=1e-08, decay=0.0), weighted_metrics=[])

call = [tf.keras.callbacks.EarlyStopping(patience=50,restore_best_weights=True)]

history_1 = vae_1.fit(X_train_1, nCW_train_1, 
                  epochs=5000, 
                  batch_size=4096, 
                  callbacks=call, 
                  validation_data = (X_val_1, nCW_val_1))

history_2 = vae_2.fit(X_train_2, nCW_train_2, 
                  epochs=5000, 
                  batch_size=4096, 
                  callbacks=call, 
                  validation_data = (X_val_2, nCW_val_2))

history_3 = vae_3.fit(X_train_3, nCW_train_3, 
                  epochs=5000, 
                  batch_size=4096, 
                  callbacks=call, 
                  validation_data = (X_val_3, nCW_val_3))

history_4 = vae_4.fit(X_train_4, nCW_train_4, 
                  epochs=5000, 
                  batch_size=4096, 
                  callbacks=call, 
                  validation_data = (X_val_4, nCW_val_4))


#--------------------
# SPLIT MORE DATASETS
#--------------------


# INPUTS
X_bg = BG.loc[:,['normalisedCombinedWeight','MET', 'MET_Phi','jet_pt', 'jet_e', 'jet_eta', 'jet_phi', 'ljet_pt', 'topjet_pt',
       'ljet_e', 'topjet_e', 'ljet_m', 'topjet_m', 'ljet_eta', 'topjet_eta',
       'ljet_phi', 'topjet_phi', 'Omega', 'HT', 'Centrality',
       'DeltaR_max', 'jet_DL1r_max', 'jet_px', 'jet_py',
       'jet_pz', 'ljet_px', 'ljet_py', 'ljet_pz', 'MET_m', 'MET_eta', 'MET_px',
       'MET_py','gen_split','train_weight','LABEL']]

# WEIGHTS OF bg_test
nCW_bg_test = (X_bg.loc[X_bg['gen_split'] == 'test'])['normalisedCombinedWeight']

# bg_test TO JOIN TO signals_test TO TEST AUTOENCODER
bg_train = (X_bg.loc[X_bg['gen_split'] == 'train']).drop(columns=['normalisedCombinedWeight',
                                                                'gen_split','train_weight','LABEL'])

# bg_test TO JOIN TO signals_test TO TEST AUTOENCODER
bg_test = (X_bg.loc[X_bg['gen_split'] == 'test']).drop(columns=['normalisedCombinedWeight',
                                                                'gen_split','train_weight','LABEL'])
# DEFINE LABELS TO USE FOR THE ROC CURVE
y_bg = BG.loc[:,['gen_split','LABEL']]

y_bg_test = (y_bg.loc[y_bg['gen_split'] == 'test']).drop(columns=['gen_split'])

#

# INPUTS
X_s1 = s1.loc[:,['normalisedCombinedWeight','MET', 'MET_Phi','jet_pt', 'jet_e', 'jet_eta', 'jet_phi', 'ljet_pt', 'topjet_pt',
       'ljet_e', 'topjet_e', 'ljet_m', 'topjet_m', 'ljet_eta', 'topjet_eta',
       'ljet_phi', 'topjet_phi', 'Omega', 'HT', 'Centrality',
       'DeltaR_max', 'jet_DL1r_max', 'jet_px', 'jet_py',
       'jet_pz', 'ljet_px', 'ljet_py', 'ljet_pz', 'MET_m', 'MET_eta', 'MET_px',
       'MET_py','gen_split','train_weight','LABEL']]

# WEIGHTS OF signals_test
nCW_s1_test = (X_s1.loc[X_s1['gen_split'] == 'test'])['normalisedCombinedWeight']


# signals_test TO JOIN TO bg_test TO TEST AUTOENCODER
s1_test = (X_s1.loc[X_s1['gen_split'] == 'test']).drop(columns=['normalisedCombinedWeight',
                                                              'gen_split','train_weight','LABEL'])

# DEFINE LABELS TO USE FOR THE ROC CURVE
y_s1 = s1.loc[:,['gen_split','LABEL']]

y_s1_test = (y_s1.loc[y_s1['gen_split'] == 'test']).drop(columns=['gen_split'])

#

# INPUTS
X_s2 = s2.loc[:,['normalisedCombinedWeight','MET', 'MET_Phi','jet_pt', 'jet_e', 'jet_eta', 'jet_phi', 'ljet_pt', 'topjet_pt',
       'ljet_e', 'topjet_e', 'ljet_m', 'topjet_m', 'ljet_eta', 'topjet_eta',
       'ljet_phi', 'topjet_phi', 'Omega', 'HT', 'Centrality',
       'DeltaR_max', 'jet_DL1r_max', 'jet_px', 'jet_py',
       'jet_pz', 'ljet_px', 'ljet_py', 'ljet_pz', 'MET_m', 'MET_eta', 'MET_px',
       'MET_py','gen_split','train_weight','LABEL']]

# WEIGHTS OF signals_test
nCW_s2_test = (X_s2.loc[X_s2['gen_split'] == 'test'])['normalisedCombinedWeight']


# signals_test TO JOIN TO bg_test TO TEST AUTOENCODER
s2_test = (X_s2.loc[X_s2['gen_split'] == 'test']).drop(columns=['normalisedCombinedWeight',
                                                              'gen_split','train_weight','LABEL'])

# DEFINE LABELS TO USE FOR THE ROC CURVE
y_s2 = s2.loc[:,['gen_split','LABEL']]

y_s2_test = (y_s2.loc[y_s2['gen_split'] == 'test']).drop(columns=['gen_split'])

#

# INPUTS
X_s3 = s3.loc[:,['normalisedCombinedWeight','MET', 'MET_Phi','jet_pt', 'jet_e', 'jet_eta', 'jet_phi', 'ljet_pt', 'topjet_pt',
       'ljet_e', 'topjet_e', 'ljet_m', 'topjet_m', 'ljet_eta', 'topjet_eta',
       'ljet_phi', 'topjet_phi', 'Omega', 'HT', 'Centrality',
       'DeltaR_max', 'jet_DL1r_max', 'jet_px', 'jet_py',
       'jet_pz', 'ljet_px', 'ljet_py', 'ljet_pz', 'MET_m', 'MET_eta', 'MET_px',
       'MET_py','gen_split','train_weight','LABEL']]

# WEIGHTS OF signals_test
nCW_s3_test = (X_s3.loc[X_s3['gen_split'] == 'test'])['normalisedCombinedWeight']


# signals_test TO JOIN TO bg_test TO TEST AUTOENCODER
s3_test = (X_s3.loc[X_s3['gen_split'] == 'test']).drop(columns=['normalisedCombinedWeight',
                                                              'gen_split','train_weight','LABEL'])

# DEFINE LABELS TO USE FOR THE ROC CURVE
y_s3 = s3.loc[:,['gen_split','LABEL']]

y_s3_test = (y_s3.loc[y_s3['gen_split'] == 'test']).drop(columns=['gen_split'])

#

# INPUTS
X_s4 = s4.loc[:,['normalisedCombinedWeight','MET', 'MET_Phi','jet_pt', 'jet_e', 'jet_eta', 'jet_phi', 'ljet_pt', 'topjet_pt',
       'ljet_e', 'topjet_e', 'ljet_m', 'topjet_m', 'ljet_eta', 'topjet_eta',
       'ljet_phi', 'topjet_phi', 'Omega', 'HT', 'Centrality',
       'DeltaR_max', 'jet_DL1r_max', 'jet_px', 'jet_py',
       'jet_pz', 'ljet_px', 'ljet_py', 'ljet_pz', 'MET_m', 'MET_eta', 'MET_px',
       'MET_py','gen_split','train_weight','LABEL']]

# WEIGHTS OF signals_test
nCW_s4_test = (X_s4.loc[X_s1['gen_split'] == 'test'])['normalisedCombinedWeight']


# signals_test TO JOIN TO bg_test TO TEST AUTOENCODER
s4_test = (X_s4.loc[X_s4['gen_split'] == 'test']).drop(columns=['normalisedCombinedWeight',
                                                              'gen_split','train_weight','LABEL'])

# DEFINE LABELS TO USE FOR THE ROC CURVE
y_s4 = s4.loc[:,['gen_split','LABEL']]

y_s4_test = (y_s4.loc[y_s4['gen_split'] == 'test']).drop(columns=['gen_split'])


#----------------
# STANDARD SCALER
#----------------


scaler = StandardScaler()

bg_train = pd.DataFrame(scaler.fit_transform(bg_train),columns = bg_train.columns)
bg_test = pd.DataFrame(scaler.transform(bg_test),columns = bg_test.columns)
s1_test = pd.DataFrame(scaler.transform(s1_test),columns = s1_test.columns)
s2_test = pd.DataFrame(scaler.transform(s2_test),columns = s2_test.columns)
s3_test = pd.DataFrame(scaler.transform(s3_test),columns = s3_test.columns)
s4_test = pd.DataFrame(scaler.transform(s4_test),columns = s4_test.columns)


#-------------------------------------
# MSE RECONSTRUCTION LOSS FOR SIGNAL 1
#-------------------------------------


bg_test_tensor = tf.cast(bg_test, tf.float32).numpy()
s1_test_tensor = tf.cast(s1_test, tf.float32).numpy()

reconstructions_bg = vae_1.custom_predict(bg_test_tensor)
bg_1_loss_mse = tf.keras.losses.mse(bg_test, reconstructions_bg)

reconstructions_s1 = vae_1.custom_predict(s1_test_tensor)
s1_loss_mse = tf.keras.losses.mse(s1_test, reconstructions_s1)


plt.figure()
plt.hist(bg_1_loss_mse, bins='auto', density=True, histtype='stepfilled', alpha=0.5, label='Background', color = 'b' )
plt.hist(s1_loss_mse, bins='auto', density=True, 
         histtype='step', alpha=0.5, label='ResmMed4000mX1lb0p2yp0p4', color = 'r')

plt.xlabel('Mean Squared Error (log scale)')
plt.ylabel('Occurrence')
plt.yscale('log')
plt.title('MSE reconstruction - log scale - model 1')
plt.legend()


#-------------------------------------
# MSE RECONSTRUCTION LOSS FOR SIGNAL 2
#-------------------------------------


s2_test_tensor = tf.cast(s2_test, tf.float32).numpy()

reconstructions_bg = vae_2.custom_predict(bg_test_tensor)
bg_2_loss_mse = tf.keras.losses.mse(bg_test, reconstructions_bg)

reconstructions_s2 = vae_2.custom_predict(s2_test_tensor)
s2_loss_mse = tf.keras.losses.mse(s2_test, reconstructions_s2)


plt.figure()
plt.hist(bg_2_loss_mse, bins='auto', density=True, histtype='stepfilled', alpha=0.5, label='Background', color = 'b' )
plt.hist(s2_loss_mse, bins='auto', density=True, 
         histtype='step', alpha=0.5, label='bbA2000_yb2_Zhvvbb', color = 'g')

plt.xlabel('Mean Squared Error (log scale)')
plt.ylabel('Occurrence')
plt.yscale('log')
plt.title('MSE reconstruction - log scale - model 2')
plt.legend()


#-------------------------------------
# MSE RECONSTRUCTION LOSS FOR SIGNAL 3
#-------------------------------------


s3_test_tensor = tf.cast(s3_test, tf.float32).numpy()

reconstructions_bg = vae_3.custom_predict(bg_test_tensor)
bg_3_loss_mse = tf.keras.losses.mse(bg_test, reconstructions_bg)

reconstructions_s3 = vae_3.custom_predict(s3_test_tensor)
s3_loss_mse = tf.keras.losses.mse(s3_test, reconstructions_s3)


plt.figure()
plt.hist(bg_3_loss_mse, bins='auto', density=True, histtype='stepfilled', alpha=0.5, label='Background', color = 'b' )
plt.hist(s3_loss_mse, bins='auto', density=True, 
         histtype='step', alpha=0.5, label='GG_direct_2000_0', color = 'y')

plt.xlabel('Mean Squared Error (log scale)')
plt.ylabel('Occurrence')
plt.yscale('log')
plt.title('MSE reconstruction - log scale - model 3')
plt.legend()


#-------------------------------------
# MSE RECONSTRUCTION LOSS FOR SIGNAL 4
#-------------------------------------


s4_test_tensor = tf.cast(s4_test, tf.float32).numpy()

reconstructions_bg = vae_4.custom_predict(bg_test_tensor)
bg_4_loss_mse = tf.keras.losses.mse(bg_test, reconstructions_bg)

reconstructions_s4 = vae_4.custom_predict(s4_test_tensor)
s4_loss_mse = tf.keras.losses.mse(s4_test, reconstructions_s4)


plt.figure()
plt.hist(bg_4_loss_mse, bins='auto', density=True, histtype='stepfilled', alpha=0.5, label='Background', color = 'b' )
plt.hist(s4_loss_mse, bins='auto', density=True, 
         histtype='step', alpha=0.5, label='HVT_Agv1_VzZH_vvqq_m1000', color = 'k')

plt.xlabel('Mean Squared Error (log scale)')
plt.ylabel('Occurrence')
plt.yscale('log')
plt.title('MSE reconstruction - log scale - model 4')
plt.legend()


#--------------------
# MODELS' PREDICTIONS
#--------------------


bg_s1_test = pd.concat([s1_test,bg_test], ignore_index=True)
bg_s2_test = pd.concat([s2_test,bg_test], ignore_index=True)
bg_s3_test = pd.concat([s3_test,bg_test], ignore_index=True)
bg_s4_test = pd.concat([s4_test,bg_test], ignore_index=True)

bg_s1_tensor = tf.cast(bg_s1_test, tf.float32).numpy()
bg_s2_tensor = tf.cast(bg_s2_test, tf.float32).numpy()
bg_s3_tensor = tf.cast(bg_s3_test, tf.float32).numpy()
bg_s4_tensor = tf.cast(bg_s4_test, tf.float32).numpy()

reconstructions_bg_s1 = vae_1.custom_predict(bg_s1_tensor)
preds1 = tf.keras.losses.mse(reconstructions_bg_s1, bg_s1_test)

reconstructions_bg_s2 = vae_2.custom_predict(bg_s2_tensor)
preds2 = tf.keras.losses.mse(reconstructions_bg_s2, bg_s2_test)

reconstructions_bg_s3 = vae_3.custom_predict(bg_s3_tensor)
preds3 = tf.keras.losses.mse(reconstructions_bg_s3, bg_s3_test)

reconstructions_bg_s4 = vae_4.custom_predict(bg_s4_tensor)
preds4 = tf.keras.losses.mse(reconstructions_bg_s4, bg_s4_test)


#----------
# ROC CURVE
#----------


y_1 = pd.concat([y_s1_test,y_bg_test], ignore_index=True)
y_2 = pd.concat([y_s2_test,y_bg_test], ignore_index=True)
y_3 = pd.concat([y_s3_test,y_bg_test], ignore_index=True)
y_4 = pd.concat([y_s4_test,y_bg_test], ignore_index=True)


def my_plot_roc_curve(preds1,y_1,preds2,y_2,preds3,y_3,preds4,y_4):
    
    fpr1, tpr1, _ = roc_curve(y_1, -preds1, pos_label=0)
    roc_auc1 = roc_auc_score(y_1, preds1)
    print('AUC score - ResmMed4000mX1lb0p2yp0p4: ', roc_auc1)
    
    fpr2, tpr2, _ = roc_curve(y_2, -preds2, pos_label=0)
    roc_auc2 = roc_auc_score(y_2, preds2)
    print('AUC score - bbA2000_yb2_Zhvvbb: ', roc_auc2)
    
    fpr3, tpr3, _ = roc_curve(y_3, -preds3, pos_label=0)
    roc_auc3 = roc_auc_score(y_3, preds3)
    print('AUC score - GG_direct_2000_0: ', roc_auc3)
    
    fpr4, tpr4, _ = roc_curve(y_4, -preds4, pos_label=0)
    roc_auc4 = roc_auc_score(y_4, preds4)
    print('AUC score - HVT_Agv1_VzZH_vvqq_m1000: ', roc_auc4)
    
    plt.figure()
    plt.plot(fpr1,tpr1,label='ResmMed4000mX1lb0p2yp0p4 - ' +str(np.round(roc_auc1,4)), color='r')
    plt.plot(fpr2,tpr2,label='bbA2000_yb2_Zhvvbb               - ' +str(np.round(roc_auc2,4)), color='g')
    plt.plot(fpr3,tpr3,label='GG_direct_2000_0                      - ' +str(np.round(roc_auc3,4)), color='y')
    plt.plot(fpr4,tpr4,label='HVT_Agv1_VzZH_vvqq_m1000  - ' +str(np.round(roc_auc4,4)), color='k')

    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.title('ROC curve - Autoencoder Contaminated')
    plt.legend()

from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay

my_plot_roc_curve(preds1,y_1,preds2,y_2,preds3,y_3,preds4,y_4)


#----------------------------------------------------
# SCALE -> log10(MSE(reconstructed, X_test)) -> [0,1]
#----------------------------------------------------


bg_1_loss_log = np.log10(bg_1_loss_mse)
bg_2_loss_log = np.log10(bg_2_loss_mse)
bg_3_loss_log = np.log10(bg_3_loss_mse)
bg_4_loss_log = np.log10(bg_4_loss_mse)
s1_loss_log = np.log10(s1_loss_mse)
s2_loss_log = np.log10(s2_loss_mse)
s3_loss_log = np.log10(s3_loss_mse)
s4_loss_log = np.log10(s4_loss_mse)
 
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
bg_1_loss_log = bg_1_loss_log.reshape(-1, 1)
bg_1_loss_log_scaled = scaler.fit_transform(bg_1_loss_log)

bg_2_loss_log = bg_2_loss_log.reshape(-1, 1)
bg_2_loss_log_scaled = scaler.fit_transform(bg_2_loss_log)

bg_3_loss_log = bg_3_loss_log.reshape(-1, 1)
bg_3_loss_log_scaled = scaler.fit_transform(bg_3_loss_log)

bg_4_loss_log = bg_4_loss_log.reshape(-1, 1)
bg_4_loss_log_scaled = scaler.fit_transform(bg_4_loss_log)


s1_loss_log = s1_loss_log.reshape(-1, 1)
s1_loss_log_scaled = scaler.fit_transform(s1_loss_log)

s2_loss_log = s2_loss_log.reshape(-1, 1)
s2_loss_log_scaled = scaler.fit_transform(s2_loss_log)

s3_loss_log = s3_loss_log.reshape(-1, 1)
s3_loss_log_scaled = scaler.fit_transform(s3_loss_log)

s4_loss_log = s4_loss_log.reshape(-1, 1)
s4_loss_log_scaled = scaler.fit_transform(s4_loss_log)


import csv

with open('VAEcont_1.txt', 'w') as f:
    csv.writer(f, delimiter=' ').writerows(np.concatenate((s1_loss_log_scaled,bg_1_loss_log_scaled)))

with open('VAEcont_2.txt', 'w') as f:
    csv.writer(f, delimiter=' ').writerows(np.concatenate((s2_loss_log_scaled,bg_2_loss_log_scaled)))

with open('VAEcont_3.txt', 'w') as f:
    csv.writer(f, delimiter=' ').writerows(np.concatenate((s3_loss_log_scaled,bg_3_loss_log_scaled)))

with open('VAEcont_4.txt', 'w') as f:
    csv.writer(f, delimiter=' ').writerows(np.concatenate((s4_loss_log_scaled,bg_4_loss_log_scaled)))


#---------------
# MODELS' SCORES
#---------------


def hist(bg_1_loss_log_scaled,s1_loss_log_scaled,nCW_bg_test,nCW_s1_test):
    plt.figure()
    plt.hist(bg_1_loss_log_scaled,
           color='b', alpha=0.5, 
           bins=100,
           histtype='stepfilled', density=True,
           label='Background', weights=nCW_bg_test)
    plt.hist(s1_loss_log_scaled,
           color='r', alpha=0.5,
           bins=100,
           histtype='stepfilled', density=True,
          label='ResmMed4000mX1lb0p2yp0p4', weights=nCW_s1_test)
    plt.legend()
    plt.title("AE contaminated model 1 score")
    plt.xlabel('Anomaly Score')
    plt.ylabel('Occurrences')
    plt.xlim([0,1])
    plt.savefig('/Users/cadodo/Desktop/VAEcont_model1_score')

hist(bg_1_loss_log_scaled,s1_loss_log_scaled,nCW_bg_test,nCW_s1_test)


def hist(bg_2_loss_log_scaled,s2_loss_log_scaled,nCW_bg_test,nCW_s2_test):
    plt.figure()
    plt.hist(bg_2_loss_log_scaled,
           color='b', alpha=0.5, 
           bins=100,
           histtype='stepfilled', density=True,
           label='Background', weights=nCW_bg_test)
    plt.hist(s2_loss_log_scaled,
           color='g', alpha=0.5,
           bins=100,
           histtype='stepfilled', density=True,
          label='bbA2000_yb2_Zhvvbb', weights=np.absolute(nCW_s2_test))
    plt.legend()
    plt.title("AE contaminated model 2 score")
    plt.xlabel('Anomaly Score')
    plt.ylabel('Occurrences')
    plt.xlim([0,1])
    plt.savefig('/Users/cadodo/Desktop/VAEcont_model2_score')

hist(bg_2_loss_log_scaled,s2_loss_log_scaled,nCW_bg_test,nCW_s2_test)


def hist(bg_3_loss_log_scaled,s3_loss_log_scaled,nCW_bg_test,nCW_s3_test):
    plt.figure()
    plt.hist(bg_3_loss_log_scaled,
           color='b', alpha=0.5, 
           bins=100,
           histtype='stepfilled', density=True,
           label='Background', weights=nCW_bg_test)
    plt.hist(s3_loss_log_scaled,
           color='y', alpha=0.5,
           bins=100,
           histtype='stepfilled', density=True,
          label='GG_direct_2000_0', weights=nCW_s3_test)
    plt.legend()
    plt.title("AE contaminated model 3 score")
    plt.xlabel('Anomaly Score')
    plt.ylabel('Occurrences')
    plt.xlim([0,1])
    plt.savefig('/Users/cadodo/Desktop/VAEcont_model3_score')

hist(bg_3_loss_log_scaled,s3_loss_log_scaled,nCW_bg_test,nCW_s3_test)


def hist(bg_4_loss_log_scaled,s4_loss_log_scaled,nCW_bg_test,nCW_s4_test):
    plt.figure()
    plt.hist(bg_4_loss_log_scaled,
           color='b', alpha=0.5, 
           bins=100,
           histtype='stepfilled', density=True,
           label='Background', weights=nCW_bg_test)
    plt.hist(s4_loss_log_scaled,
           color='k', alpha=0.5,
           bins=100,
           histtype='stepfilled', density=True,
          label='HVT_Agv1_VzZH_vvqq_m1000')
    plt.legend()
    plt.title("AE contaminated model 4 score")
    plt.xlabel('Anomaly Score')
    plt.ylabel('Occurrences')
    plt.xlim([0,1])
    plt.savefig('/Users/cadodo/Desktop/VAEcont_model4_score')

hist(bg_4_loss_log_scaled,s4_loss_log_scaled,nCW_bg_test,nCW_s4_test)