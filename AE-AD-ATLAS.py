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


# IMPORT DATASETS

signals = pd.read_csv(r'/Users/cadodo/Desktop/main/LIP/CSV/ResmMed4000mX1lb0p2yp0p4.csv')
BG = pd.read_csv(r'/Users/cadodo/Desktop/main/LIP/CSV/bkg.csv')


# ADD LABEL

BG.insert(95, "LABEL", 0, True)
signals.insert(95, "LABEL", 1, True)


# DEFINE BG TRAIN, TEST, VAL

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


# DEFINE SIGNALS TRAIN, TEST, VAL

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


# DEFINE WEIGHTS

# BG WEIGHTS
weight_bg_train = (X_bg.loc[X_bg['gen_split'] == 'train'])['train_weight'] 
weight_bg_test = (X_bg.loc[X_bg['gen_split'] == 'test'])['train_weight']
weight_bg_val = (X_bg.loc[X_bg['gen_split'] == 'val'])['train_weight']

# SIGNALS WEIGHTS
weight_signals_train = (X_signals.loc[X_signals['gen_split'] == 'train'])['train_weight']
weight_signals_test = (X_signals.loc[X_signals['gen_split'] == 'test'])['train_weight']
weight_signals_val = (X_signals.loc[X_signals['gen_split'] == 'val'])['train_weight']


# SUM WEIGHTS (verification)

class_weights_train = (weight_bg_train.values.sum(),weight_signals_train.values.sum())    # = (0.99, 1)
class_weights_test = (weight_bg_test.values.sum(),weight_signals_test.values.sum())       # = (0.99, 1)
class_weights_val = (weight_bg_val.values.sum(),weight_signals_val.values.sum())          # = (1, 1)


# JOIN DATASETS

# JOIN NECESSARY DATASETS FOR X
X_train = bg_train
X_test = pd.concat([signals_test, bg_test], ignore_index=True)
X_val = bg_val

# JOIN NECESSARY DATASETS FOR y
y_test = pd.concat([y_signals_test, y_bg_test], ignore_index=True).values
y_val = y_bg_val

# JOIN NECESSARY DATASETS FOR weight
weight_train = weight_bg_train.values
weight_test = pd.concat([weight_signals_test, weight_bg_test], ignore_index=True).values.reshape(-1,1)
weight_val = weight_bg_val
nCW_test = pd.concat([nCW_signals_test, nCW_bg_test], ignore_index=True).values
nCW_test = nCW_test.reshape(-1,1)


# DEFINE AUTOENCODER ARCHITECTURE (optimized for latent space of 2)

class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(X_train.shape[1], activation=None, input_shape=(X_train.shape[1],)),
      layers.Dense(343, activation="swish"),
      layers.Dropout(0.5),
      layers.Dense(176, activation="swish"),
      layers.Dense(74, activation="swish"),
      layers.Dense(22, activation="swish"),
      layers.Dense(2, activation="swish")])

    self.decoder = tf.keras.Sequential([
      layers.Dense(2, activation="swish"),
      layers.Dense(22, activation="swish"),
      layers.Dense(74, activation="swish"),
      layers.Dense(176, activation="swish"),
      layers.Dropout(0.5),
      layers.Dense(343, activation="swish"),
      layers.Dense(X_train.shape[1], activation=None)]) # N° of neurons = X's columns

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = AnomalyDetector()

autoencoder.compile(optimizer='adam',loss='mse',weighted_metrics=[tf.keras.metrics.MeanSquaredError()])


# MODEL FIT

import keras

history = autoencoder.fit(X_train, X_train, 
          epochs=1, 
          batch_size=2048,
          validation_data=(X_val, X_val, weight_val), 
          sample_weight=weight_train,
          callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
          shuffle=True)


# STANDARDISATION OF INPUTS

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


# MODEL FIT 2

history = autoencoder.fit(X_train, X_train, 
          epochs=100, 
          batch_size=2048,
          validation_data=(X_val, X_val, weight_val), 
          sample_weight=weight_train,
          callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
          shuffle=True)


# PLOT HISTOGRAMS

# DEFINE ENCODED AND DECODED DATA BY THE AUTOENCODER
bg_test_tensor = tf.cast(bg_test, tf.float16).numpy()
signals_test_tensor = tf.cast(signals_test, tf.float16).numpy()

encoded_bg = autoencoder.encoder(bg_test_tensor).numpy()
decoded_bg = autoencoder.decoder(encoded_bg).numpy()
 
encoded_signals = autoencoder.encoder(signals_test_tensor).numpy()
decoded_signals = autoencoder.decoder(encoded_signals).numpy()

 
inputs = ['MET_px', 'MET_py','jet_e', 'jet_px', 'jet_py', 'jet_pz',
            'ljet_e', 'ljet_px', 'ljet_py','ljet_pz','HT','jet_DL1r_max']
xlim = [[-4000,4000],[-4000,4000],[0,5000],[-4000,4000],[-4000,4000],[-5000,5000],[0,5000],[-4000,4000],
        [-4000,4000],[-5000,5000],[0,7000],[-11,16]]
ylim = [[0,0.00225],[0,0.00225],[0,0.0025],[0,0.0018],[0,0.0020],[0,0.0014],[0,0.0025],[0,0.0018],[0,0.002],
        [0,0.0014],[0,0.003],[0,0.3]]
bins = 200
dens = True
 
 
def hist(inputs,xlim,ylim,bins,dens,bg_test_tensor,signals_test_tensor,decoded_bg,decoded_signals,nCW_bg_test,nCW_signals_test):
    for j in range(len(bg_test_tensor[0])):
        array_bg = []
        array_signals = []
        for i in range(len(bg_test_tensor)):
            array_bg.append(decoded_bg[i][j])
        for k in range(len(signals_test_tensor)):
            array_signals.append(decoded_signals[k][j])
        plt.figure()
        plt.hist(bg_test[inputs[j]], bins=bins, density=dens, histtype='stepfilled', alpha=0.7,
                weights=nCW_bg_test, color='#7f7fff', label='bg_test INPUT')
        plt.hist(array_bg, bins=bins, density=dens, histtype='step', alpha=1,
                weights=nCW_bg_test, color='#5656ff', label='bg_test OUTPUT')    
        plt.hist(signals_test[inputs[j]], bins=bins, density=dens, histtype='stepfilled', alpha=0.7,
                weights=nCW_signals_test, color='#ff7f7f', label='signals_test INPUT')
        plt.hist(array_signals, bins=bins, density=dens, histtype='step', alpha=1,
                weights=nCW_signals_test, color='#ff4444', label='signals_test OUTPUT')
    
        plt.title('All Features - ' + str(inputs[j]))
        plt.xlim(xlim[j])
        plt.ylim(ylim[j])
        plt.legend()
        plt.xlabel(str(inputs[j])+' (GeV)')
        plt.ylabel('Occurrence Density')
        plt.savefig('All features - ' + str(inputs[j]))
 
hist(inputs,xlim,ylim,bins,dens,bg_test_tensor,signals_test_tensor,decoded_bg,decoded_signals,nCW_bg_test,nCW_signals_test)


# CALCULATE ERRORS

bg_std = X_test[y_test==0]       # standardised
signals_std = X_test[y_test==1]  # standardised

# Background reconstruction
reconstructions_bg = autoencoder.predict(bg_std)
bg_loss = tf.keras.losses.mae(reconstructions_bg, bg_std)

# Signals reconstruction
reconstructions_signals = autoencoder.predict(signals_std)
signals_loss = tf.keras.losses.mae(reconstructions_signals, signals_std)


# PLOT MSE

bg_loss = bg_loss.numpy()
signals_loss = signals_loss.numpy()

hist_bg, bins_bg = np.histogram(bg_loss, bins='auto', density=True)
hist_signals, bins_signals = np.histogram(signals_loss, bins='auto', density=True)

plt.figure()
plt.hist(bg_loss, bins='auto', density=True, alpha=0.5, label='BG', color = 'b' )
plt.hist(signals_loss, bins='auto', density=True, alpha=0.5, label='Signals', color = 'r')
plt.xlabel('MSE')
plt.ylabel('Occurrence')
plt.legend()
plt.title('MSE reconstruction')
plt.savefig('MSE')

plt.figure()
plt.hist(bg_loss, bins='auto', density=True, alpha=0.5, label='BG', color = 'b' )
plt.hist(signals_loss, bins='auto', density=True, alpha=0.5, label='Signals', color = 'r')
plt.xlabel('MSE (log scale)')
plt.ylabel('Occurrence')
plt.xscale('log')
plt.title('MSE reconstruction - log scale')
plt.legend()
plt.savefig('MSE - log scale')


# CHANGE VARIABLES (anomaly scores = log10[MSE]->[0,1])

bg_loss_log = np.log10(bg_loss)
signals_loss_log = np.log10(signals_loss)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
bg_loss_log = bg_loss_log.reshape(-1, 1)
bg_loss_log_scaled = scaler.fit_transform(bg_loss_log)

signals_loss_log = signals_loss_log.reshape(-1, 1)
signals_loss_log_scaled = scaler.fit_transform(signals_loss_log)

anomaly_scores = np.concatenate((signals_loss_log_scaled.reshape(-1),bg_loss_log_scaled.reshape(-1)))


# PLOT MODEL SCORE

density=True
y_test = y_test.reshape(-1)
def model_score(preds,y_test,density,weight_test):
    plt.figure()
    plt.hist(preds[y_test == 0],
         color='b', alpha=0.5, 
         bins=200,
         histtype='stepfilled', density=density,
         label='BG (test)', weights=nCW_test[y_test == 0])
    plt.hist(preds[y_test == 1],
         color='r', alpha=0.5,
         bins=200,
         histtype='stepfilled', density=density,
         label='signals (test)', weights=nCW_test[y_test == 1])

    plt.legend()
    plt.xlabel('Score (log10(MSE))->[0,1]')
    plt.ylabel('Occurrence Density')
    plt.title("Model score - Autoencoder (bg_train only)")
    plt.savefig('AE model score')

model_score(anomaly_scores,y_test,density,weight_test)


# RECONSTRUCTIONS FOR ROC CURVE

X_test_tensor = tf.cast(X_test, tf.float16).numpy()
threshold = np.mean(bg_loss) + np.std(bg_loss)

def predict(model, data, threshold):
  reconstructions = model(data)
  loss = tf.keras.losses.mae(reconstructions, data)
  return loss

preds = predict(autoencoder, X_test_tensor, threshold).numpy()


# PLOT ROC CURVE

def my_plot_roc_curve(model, y_scores, y_test):
        
    if len(y_scores.shape) == 2:
        if y_scores.shape[1] == 1:
            y_scores = y_scores.reshape(-1)
        elif y_scores.shape[1] == 2:
            y_scores = y_scores[:,1].reshape(-1)

    fpr, tpr, _ = roc_curve(y_test, -y_scores, pos_label=0)
    roc_auc = roc_auc_score(y_test, y_scores)
    print('AUC score: ', roc_auc)
    
    plt.clf()
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=model.__class__.__name__)
    display.plot()
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.title('ROC curve - Autoencoder')
    plt.savefig('AE ROC curve')

from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay 
my_plot_roc_curve(autoencoder, preds, y_test)
