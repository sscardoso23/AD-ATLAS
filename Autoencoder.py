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


BG = pd.read_csv(r'/Users/cadodo/Desktop/main/LIP/CSV/bkg.csv')
s1 = pd.read_csv(r'/Users/cadodo/Desktop/main/LIP/CSV/ResmMed4000mX1lb0p2yp0p4.csv')
s2 = pd.read_csv(r'/Users/cadodo/Desktop/main/LIP/CSV/bbA2000_yb2_Zhvvbb.csv')
s3 = pd.read_csv(r'/Users/cadodo/Desktop/main/LIP/CSV/GG_direct_2000_0.csv')
s4 = pd.read_csv(r'/Users/cadodo/Desktop/main/LIP/CSV/HVT_Agv1_VzZH_vvqq_m1000.csv')

BG['LABEL'] = 0
s1['LABEL'] = 1
s2['LABEL'] = 1
s3['LABEL'] = 1
s4['LABEL'] = 1


#---------------
# SPLIT DATASETS
#---------------


X_bg = BG.loc[:,['normalisedCombinedWeight','MET', 'MET_Phi','jet_pt', 
                 'jet_e', 'jet_eta', 'jet_phi', 'ljet_pt', 'topjet_pt',
                 'ljet_e', 'topjet_e', 'ljet_m', 'topjet_m', 'ljet_eta', 'topjet_eta',
                 'ljet_phi', 'topjet_phi', 'Omega', 'HT', 'Centrality',
                 'DeltaR_max', 'jet_DL1r_max', 'jet_px', 'jet_py',
                 'jet_pz', 'ljet_px', 'ljet_py', 'ljet_pz', 'MET_m', 'MET_eta', 'MET_px',
                 'MET_py','gen_split','train_weight','LABEL']]

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
X_s2 = s2.loc[:,['normalisedCombinedWeight','MET', 'MET_Phi','jet_pt', 'jet_e', 
                 'jet_eta', 'jet_phi', 'ljet_pt', 'topjet_pt',
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
nCW_s4_test = (X_s4.loc[X_s4['gen_split'] == 'test'])['normalisedCombinedWeight']


# signals_test TO JOIN TO bg_test TO TEST AUTOENCODER
s4_test = (X_s4.loc[X_s4['gen_split'] == 'test']).drop(columns=['normalisedCombinedWeight',
                                                              'gen_split','train_weight','LABEL'])

# DEFINE LABELS TO USE FOR THE ROC CURVE
y_s4 = s4.loc[:,['gen_split','LABEL']]

y_s4_test = (y_s4.loc[y_s4['gen_split'] == 'test']).drop(columns=['gen_split'])


#---------------
# DEFINE WEIGHTS
#---------------


weight_bg_train = (X_bg.loc[X_bg['gen_split'] == 'train'])['train_weight'] 
weight_bg_test = (X_bg.loc[X_bg['gen_split'] == 'test'])['train_weight']
weight_bg_val = (X_bg.loc[X_bg['gen_split'] == 'val'])['train_weight']

X_train = bg_train
X_val = bg_val

y_val = y_bg_val

weight_train = weight_bg_train.values
weight_val = weight_bg_val


#----------------
# STANDARD SCALER
#----------------


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train),columns = X_train.columns)
X_val = pd.DataFrame(scaler.transform(X_val),columns = X_val.columns)

bg_train = pd.DataFrame(scaler.fit_transform(bg_train),columns = bg_train.columns)
bg_test = pd.DataFrame(scaler.transform(bg_test),columns = bg_test.columns)
s1_test = pd.DataFrame(scaler.transform(s1_test),columns = s1_test.columns)
s2_test = pd.DataFrame(scaler.transform(s2_test),columns = s2_test.columns)
s3_test = pd.DataFrame(scaler.transform(s3_test),columns = s3_test.columns)
s4_test = pd.DataFrame(scaler.transform(s4_test),columns = s4_test.columns)


#-------------------------
# AUTOENCODER ARCHITECTURE
#-------------------------


class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
      layers.Dense(16, activation='relu'),
      layers.Dense(8, activation='relu'),
      ])

    self.decoder = tf.keras.Sequential([
      layers.Dense(16, activation='relu'),
      layers.Dense(32, activation='relu'),
      layers.Dense(X_train.shape[1], activation=None)]) # N° of neurons = X's columns

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = AnomalyDetector()

autoencoder.compile(optimizer='adam',
                    loss='mse',
                    weighted_metrics=[tf.keras.metrics.MeanSquaredError()])


#----------
# MODEL FIT
#----------


history = autoencoder.fit(X_train, X_train, 
          epochs=1000, 
          batch_size=4096,
          validation_data=(X_val, X_val, weight_val), 
          sample_weight=weight_train,
          callbacks=[keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True)],
          shuffle=True)


#---------------------
# PLOT RECONSTRUCTIONS
#---------------------


# # DEFINE ENCODED AND DECODED DATA BY THE AUTOENCODER
bg_test_tensor = tf.cast(bg_test, tf.float32).numpy()
 
encoded_bg = np.array(autoencoder.encoder(bg_test_tensor))
decoded_bg = np.array(autoencoder.decoder(encoded_bg))

 
features = ['MET','MET_Phi','jet_pt','jet_e','jet_eta','jet_phi','ljet_pt', 
          'topjet_pt','ljet_e','topjet_e','ljet_m','topjet_m','ljet_eta', 
          'topjet_eta','ljet_phi','topjet_phi','Omega','HT','Centrality',
          'DeltaR_max','jet_DL1r_max','jet_px','jet_py','jet_pz','ljet_px', 
          'ljet_py','ljet_pz','MET_m','MET_eta','MET_px','MET_py']

bins = 200
dens = True
 
 
def hist(inputs,bins,dens,bg_test_tensor,decoded_bg,nCW_bg_test):
    for j in range(len(bg_test_tensor[0])):
        array_bg = []
        for i in range(len(bg_test_tensor)):
            array_bg.append(decoded_bg[i][j])
            
        plt.figure()
        plt.hist(bg_test[inputs[j]], bins=bins, density=dens, histtype='stepfilled', alpha=0.5,
                weights=nCW_bg_test, color='b', label='bg_test INPUT')
        plt.hist(array_bg, bins=500, density=dens, histtype='step', alpha=0.5,
                weights=nCW_bg_test, color='r', label='bg_test OUTPUT')
    
        plt.title('All Features - ' + str(inputs[j]))
        plt.legend()
        plt.xlabel(str(inputs[j])+' (GeV)')
        plt.ylabel('Occurrence Density')
 
hist(inputs,bins,dens,bg_test_tensor,decoded_bg,nCW_bg_test)


#--------------------------
# RECONSTRUCTION LOSS - MAE
#--------------------------


bg_test_tensor = tf.cast(bg_test, tf.float32).numpy()
s1_test_tensor = tf.cast(s1_test, tf.float32).numpy()
s2_test_tensor = tf.cast(s2_test, tf.float32).numpy()
s3_test_tensor = tf.cast(s3_test, tf.float32).numpy()
s4_test_tensor = tf.cast(s4_test, tf.float32).numpy()

def predict_mae(model, data):
    reconstructions = model(data)
    loss = tf.keras.losses.mae(reconstructions, data)
    return loss

bg_loss_mae = predict_mae(autoencoder, bg_test_tensor)
s1_loss_mae = predict_mae(autoencoder, s1_test_tensor)
s2_loss_mae = predict_mae(autoencoder, s2_test_tensor)
s3_loss_mae = predict_mae(autoencoder, s3_test_tensor)
s4_loss_mae = predict_mae(autoencoder, s4_test_tensor)

bg_loss_mae = bg_loss_mae.numpy()
s1_loss_mae = s1_loss_mae.numpy()
s2_loss_mae = s2_loss_mae.numpy()
s3_loss_mae = s3_loss_mae.numpy()
s4_loss_mae = s4_loss_mae.numpy()

plt.figure()
plt.hist(bg_loss_mae, bins='auto', density=True, histtype='stepfilled', alpha=0.5, label='Background', color = 'b' )
plt.hist(s1_loss_mae, bins='auto', density=True, 
         histtype='step', alpha=0.5, label='ResmMed4000mX1lb0p2yp0p4', color = 'r')
plt.hist(s2_loss_mae, bins='auto', density=True, 
         histtype='step', alpha=0.5, label='bbA2000_yb2_Zhvvbb', color = 'g')
plt.hist(s3_loss_mae, bins='auto', density=True, 
         histtype='step', alpha=0.5, label='GG_direct_2000_0', color = 'y')
plt.hist(s4_loss_mae, bins='auto', density=True, 
         histtype='step', alpha=0.5, label='HVT_Agv1_VzZH_vvqq_m1000', color = 'k')
plt.xlabel('Mean Absolute Error (log scale)')
plt.ylabel('Occurrence')
plt.yscale('log')
plt.title('MAE reconstruction - log scale')
plt.legend()


#--------------------------
# RECONSTRUCTION LOSS - MSE
#--------------------------


def predict_mse(model, data):
    reconstructions = model(data)
    loss = tf.keras.losses.mse(reconstructions, data)
    return loss

bg_loss_mse = predict_mse(autoencoder, bg_test_tensor)
s1_loss_mse = predict_mse(autoencoder, s1_test_tensor)
s2_loss_mse = predict_mse(autoencoder, s2_test_tensor)
s3_loss_mse = predict_mse(autoencoder, s3_test_tensor)
s4_loss_mse = predict_mse(autoencoder, s4_test_tensor)

bg_loss_mse = bg_loss_mse.numpy()
s1_loss_mse = s1_loss_mse.numpy()
s2_loss_mse = s2_loss_mse.numpy()
s3_loss_mse = s3_loss_mse.numpy()
s4_loss_mse = s4_loss_mse.numpy()

plt.figure()
plt.hist(bg_loss_mse, bins='auto', density=True, histtype='stepfilled', alpha=0.5, label='Background', color = 'b' )
plt.hist(s1_loss_mse, bins='auto', density=True, 
         histtype='step', alpha=0.5, label='ResmMed4000mX1lb0p2yp0p4', color = 'r')
plt.hist(s2_loss_mse, bins='auto', density=True, 
         histtype='step', alpha=0.5, label='bbA2000_yb2_Zhvvbb', color = 'g')
plt.hist(s3_loss_mse, bins='auto', density=True, 
         histtype='step', alpha=0.5, label='GG_direct_2000_0', color = 'y')
plt.hist(s4_loss_mse, bins='auto', density=True, 
         histtype='step', alpha=0.5, label='HVT_Agv1_VzZH_vvqq_m1000', color = 'k')
plt.xlabel('Mean Squared Error (log scale)')
plt.ylabel('Occurrence')
plt.yscale('log')
plt.title('MSE reconstruction - log scale')
plt.legend()


#---------------
# MODEL PREDICTS
#---------------


bg_s1_test = pd.concat([s1_test,bg_test], ignore_index=True)
bg_s2_test = pd.concat([s2_test,bg_test], ignore_index=True)
bg_s3_test = pd.concat([s3_test,bg_test], ignore_index=True)
bg_s4_test = pd.concat([s4_test,bg_test], ignore_index=True)

bg_s1_tensor = tf.cast(bg_s1_test, tf.float32).numpy()
bg_s2_tensor = tf.cast(bg_s2_test, tf.float32).numpy()
bg_s3_tensor = tf.cast(bg_s3_test, tf.float32).numpy()
bg_s4_tensor = tf.cast(bg_s4_test, tf.float32).numpy()


def predict(model, data):
    reconstructions = model(data)
    loss = tf.keras.losses.mse(reconstructions, data)
    return loss

preds1 = predict(autoencoder, bg_s1_tensor).numpy()
preds2 = predict(autoencoder, bg_s2_tensor).numpy()
preds3 = predict(autoencoder, bg_s3_tensor).numpy()
preds4 = predict(autoencoder, bg_s4_tensor).numpy()


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
    plt.title('ROC curve - Variational Autoencoder')
    plt.legend()
    plt.xlabel('False Positives Rate')
    plt.ylabel('True Positives Rate')

from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay

my_plot_roc_curve(preds1,y_1,preds2,y_2,preds3,y_3,preds4,y_4)


#----------------------------------------------------
# SCALE -> log10(MSE(reconstructed, X_test)) -> [0,1]
#----------------------------------------------------


bg_loss_log = np.log10(bg_loss_mse)
s1_loss_log = np.log10(s1_loss_mse)
s2_loss_log = np.log10(s2_loss_mse)
s3_loss_log = np.log10(s3_loss_mse)
s4_loss_log = np.log10(s4_loss_mse)
 
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
bg_loss_log = bg_loss_log.reshape(-1, 1)
bg_loss_log_scaled = scaler.fit_transform(bg_loss_log)

s1_loss_log = s1_loss_log.reshape(-1, 1)
s1_loss_log_scaled = scaler.fit_transform(s1_loss_log)

s2_loss_log = s2_loss_log.reshape(-1, 1)
s2_loss_log_scaled = scaler.fit_transform(s2_loss_log)

s3_loss_log = s3_loss_log.reshape(-1, 1)
s3_loss_log_scaled = scaler.fit_transform(s3_loss_log)

s4_loss_log = s4_loss_log.reshape(-1, 1)
s4_loss_log_scaled = scaler.fit_transform(s4_loss_log)


import csv

with open('AE_1.txt', 'w') as f:
    csv.writer(f, delimiter=' ').writerows(np.concatenate((s1_loss_log_scaled,bg_loss_log_scaled)))

with open('AE_2.txt', 'w') as f:
    csv.writer(f, delimiter=' ').writerows(np.concatenate((s2_loss_log_scaled,bg_loss_log_scaled)))

with open('AE_3.txt', 'w') as f:
    csv.writer(f, delimiter=' ').writerows(np.concatenate((s3_loss_log_scaled,bg_loss_log_scaled)))

with open('AE_4.txt', 'w') as f:
    csv.writer(f, delimiter=' ').writerows(np.concatenate((s4_loss_log_scaled,bg_loss_log_scaled)))


#----------------------------
# MODEL SCORE FOR EACH SIGNAL
# ---------------------------


def hist(bg_loss_log_scaled,s1_loss_log_scaled,nCW_bg_test,nCW_s1_test):
    plt.figure()
    plt.hist(bg_loss_log_scaled,
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
    plt.title("VAE model score 1")
    plt.xlabel('Anomaly Score')
    plt.ylabel('Occurrences')
    plt.xlim([0,1])

hist(bg_loss_log_scaled,s1_loss_log_scaled,nCW_bg_test,nCW_s1_test)


def hist(bg_loss_log_scaled,s2_loss_log_scaled,nCW_bg_test,nCW_s2_test):
    plt.figure()
    plt.hist(bg_loss_log_scaled,
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
    plt.title("VAE model score 2")
    plt.xlabel('Anomaly Score')
    plt.ylabel('Occurrences')
    plt.xlim([0,1])

hist(bg_loss_log_scaled,s2_loss_log_scaled,nCW_bg_test,nCW_s2_test)


def hist(bg_loss_log_scaled,s3_loss_log_scaled,nCW_bg_test,nCW_s3_test):
    plt.figure()
    plt.hist(bg_loss_log_scaled,
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
    plt.title("VAE model score 3")
    plt.xlabel('Anomaly Score')
    plt.ylabel('Occurrences')
    plt.xlim([0,1])

hist(bg_loss_log_scaled,s3_loss_log_scaled,nCW_bg_test,nCW_s3_test)


def hist(bg_loss_log_scaled,s4_loss_log_scaled,nCW_bg_test,nCW_s4_test):
    plt.figure()
    plt.hist(bg_loss_log_scaled,
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
    plt.title("VAE model score 4")
    plt.xlabel('Anomaly Score')
    plt.ylabel('Occurrences')
    plt.xlim([0,1])

hist(bg_loss_log_scaled,s4_loss_log_scaled,nCW_bg_test,nCW_s4_test)