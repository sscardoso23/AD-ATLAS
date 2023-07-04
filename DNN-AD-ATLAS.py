import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import os
from sklearn import datasets, metrics, model_selection, svm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
np.random.seed(31415)



# IMPORTING SIGNALS AND ADDING LABEL

signals = pd.read_csv(r'/home/scardoso/ResmMed4000mX1lb0p2yp0p4.csv')  # import file from pc
signals["LABEL"] = 1 # Add the LABEL 1 to the signals (last column)

# IMPORTING BACKGROUND AND ADDING LABEL

BG = pd.read_csv(r'/home/scardoso/bkg.csv')  # import file from pc
BG["LABEL"] = 0       # Add the LABEL 0 to the background (last column)


# JOINING THE 2 DATASETS

allData = pd.concat([signals, BG], ignore_index=True)



# TRAINING, SPLITTING, TESTING

X = allData.loc[:,['normalisedCombinedWeight','MET_px', 'MET_py','jet_e', 'jet_px', 'jet_py', 'jet_pz',
                  'ljet_e', 'ljet_px', 'ljet_py','ljet_pz','HT','gen_split','train_weight','LABEL']]     # Selecting features

nCW_X_test = (X.loc[X['gen_split'] == 'test'])['normalisedCombinedWeight'].values     # Creating an array for 'normalisedCombinedWeight'
nCW_X_test = nCW_X_test.reshape(-1,1)

X_train = (X.loc[X['gen_split'] == 'train']).drop(columns=['normalisedCombinedWeight','gen_split','train_weight','LABEL'])   
X_test = (X.loc[X['gen_split'] == 'test']).drop(columns=['normalisedCombinedWeight','gen_split','train_weight','LABEL'])
X_val = (X.loc[X['gen_split'] == 'val']).drop(columns=['normalisedCombinedWeight','gen_split','train_weight','LABEL'])


y = allData.loc[:,['gen_split','LABEL']]     # Selecting LABEL

y_train = (y.loc[X['gen_split'] == 'train']).drop(columns=['gen_split'])
y_test = (y.loc[X['gen_split'] == 'test']).drop(columns=['gen_split'])
y_val = (y.loc[X['gen_split'] == 'val']).drop(columns=['gen_split'])


# WEIGHTS

weight_train = (X.loc[X['gen_split'] == 'train'])[['train_weight','LABEL']]    # Weights for training
weight_val = (X.loc[X['gen_split'] == 'val'])[['train_weight','LABEL']]        # Weights for validation
weight_test = (X.loc[X['gen_split'] == 'test'])[['train_weight','LABEL']]      # Weights for testing
# all have 'train_weight' and 'LABEL'


# Calculate class_weights_train 

class_weights_train = ((weight_train.loc[weight_train['LABEL'] == 0])['train_weight'].sum(),
                       (weight_train.loc[weight_train['LABEL'] == 1])['train_weight'].sum())

print ("class_weights_train (BG, signals):",class_weights_train)  # = (0,999...,1)

class_weights_val = ((weight_val.loc[weight_val['LABEL'] == 0])['train_weight'].sum(),
                       (weight_val.loc[weight_val['LABEL'] == 1])['train_weight'].sum())

print ("class_weights_val (BG, signals):",class_weights_val)      # = (0,999...,1)


class_weights_test = ((weight_test.loc[weight_test['LABEL'] == 0])['train_weight'].sum(),
                       (weight_test.loc[weight_test['LABEL'] == 1])['train_weight'].sum())

print ("class_weights_test (BG, signals):",class_weights_test)      # = (0,999...,1)


# CREATING DNN

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow import keras


tf.random.set_seed(1234) # To have reproducible networks


model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)), # 1st hidden layer
  tf.keras.layers.Dense(1,activation="sigmoid") # output layer
])

model.compile(loss="binary_crossentropy", optimizer="adam", 
              weighted_metrics=['accuracy', keras.metrics.AUC(name="auc")])

history = model.fit(X_train, y_train.values,
                    epochs=1,
                    validation_data=(X_val, y_val, weight_val['train_weight']),
                    batch_size=1024,
                    sample_weight=weight_train['train_weight'].values,
                    callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])


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

tf.random.set_seed(1234) # to have reproducible networks


model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)), # 1st hidden layer
  tf.keras.layers.Dense(1,activation="sigmoid") # output layer
])

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              weighted_metrics=['accuracy', keras.metrics.AUC(name="auc")])

history = model.fit(X_train, y_train.values,
                    epochs=100,
                    #validation_split=0.2,   # to be used with train/test split
                    validation_data=(X_val, y_val, weight_val['train_weight']),
                    batch_size=1024,
                    sample_weight=weight_train['train_weight'].values,
                    callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])


# PLOT HISTOGRAM DNN

def hist_log(pred,test,weight):
  plt.figure()
  plt.hist(pred[test == 0],
           color='b', alpha=0.5, 
           bins=50,
           histtype='stepfilled', density=True,
           label='BG (test)', weights=weight[test == 0])

  plt.hist(pred[test == 1],
           color='r', alpha=0.5,
           bins=50,
           histtype='stepfilled', density=True,
           label='signals (test)', weights=weight[test == 1])
  plt.legend()
  plt.title("NN model score - logarithmic scale")
  plt.yscale('log')
  plt.savefig("NN model score - log scale")
  plt.show()


def hist(pred,test,weight):
  plt.figure()
  plt.hist(pred[test == 0],
           color='b', alpha=0.5, 
           bins=50,
           histtype='stepfilled', density=True,
           label='BG (test)', weights=weight[test == 0])
  plt.hist(pred[test == 1],
           color='r', alpha=0.5,
           bins=50,
           histtype='stepfilled', density=True,
          label='signals (test)', weights=weight[test == 1])
  plt.legend()
  plt.title("NN model score")
  plt.savefig("NN model score")
  plt.show()


# Make inputs of histogram the same type

y_pred_model = model.predict(X_test).reshape(-1,1)
y_test = y_test.values

# Plot the histograms
hist(y_pred_model,y_test,nCW_X_test)
hist_log(y_pred_model,y_test,nCW_X_test)


# ROC CURVE

def my_plot_roc_curve(model, X_test, y_test):
  if hasattr(model, "predict_proba"):
    y_scores = model.predict_proba(X_test)
  else:
    y_scores = model.predict(X_test)

  if len(y_scores.shape) == 2:
    if y_scores.shape[1] == 1:
      y_scores = y_scores.reshape(-1)
    elif y_scores.shape[1] == 2:
      y_scores = y_scores[:,1].reshape(-1)
  fpr, tpr, _ = roc_curve(y_test, y_scores)
  roc_auc = roc_auc_score(y_test, y_scores)
  plt.clf()
  display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=model.__class__.__name__)
  display.plot()
  plt.plot([0, 1], [0, 1], color='black', linestyle='--')
  plt.title("ROC curve - NN model")
  plt.show()
  plt.savefig("ROC curve")

# Plot the ROC curve
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay 
my_plot_roc_curve(model, X_test, y_test)


# DNN VALIDATION

fig,ax=plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(history.history['loss'],label="Training loss")
ax[0].plot(history.history['val_loss'],label="Validation loss")
ax[0].set_xlabel("Epoch")
ax[0].legend(loc='best')
plt.title("Training loss vs Validation loss")

ax[1].plot(history.history['auc'],label="Training AUC")
ax[1].plot(history.history['val_auc'],label="Validation AUC")
ax[1].set_xlabel("Epoch")
ax[1].legend(loc='best')
plt.title("Training AUC vs Validation AUC")
plt.savefig("Validation")


# SAVING THE MODEL

directory = "saved_model"

parent_dir = "/home/scardoso/"

path = os.path.join(parent_dir, directory)
os.mkdir(path)

model.save(f'/home/scardoso/saved_model/NN',save_format="tf")


# LOAD THE MODEL

from tensorflow.keras.models import load_model

new_model = tf.models.load_model(f"/home/scardoso/saved_model/ad_atlas_model", compile=False)
new_model.summary()