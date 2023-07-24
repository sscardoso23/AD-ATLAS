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


#---------------
# SPLIT DATASETS
#---------------


bg = BG.loc[:,['normalisedCombinedWeight','MET', 'MET_Phi','jet_pt', 
                 'jet_e', 'jet_eta', 'jet_phi', 'ljet_pt', 'topjet_pt',
                 'ljet_e', 'topjet_e', 'ljet_m', 'topjet_m', 'ljet_eta', 'topjet_eta',
                 'ljet_phi', 'topjet_phi', 'Omega', 'HT', 'Centrality',
                 'DeltaR_max', 'jet_DL1r_max', 'jet_px', 'jet_py',
                 'jet_pz', 'ljet_px', 'ljet_py', 'ljet_pz', 'MET_m', 'MET_eta', 'MET_px',
                 'MET_py','gen_split','train_weight','LABEL']]

bg_test = (bg.loc[bg['gen_split']=='test']).drop(columns=['normalisedCombinedWeight','gen_split','train_weight','LABEL'])
bg_train = (bg.loc[bg['gen_split']=='train']).drop(columns=['normalisedCombinedWeight','gen_split','train_weight','LABEL'])

y_bg_test = (bg.loc[bg['gen_split']=='test'])['LABEL']

nCW_bg = (bg.loc[bg['gen_split']=='test'])['normalisedCombinedWeight']

#

s_1 = s1.loc[:,['normalisedCombinedWeight','MET', 'MET_Phi','jet_pt', 
                 'jet_e', 'jet_eta', 'jet_phi', 'ljet_pt', 'topjet_pt',
                 'ljet_e', 'topjet_e', 'ljet_m', 'topjet_m', 'ljet_eta', 'topjet_eta',
                 'ljet_phi', 'topjet_phi', 'Omega', 'HT', 'Centrality',
                 'DeltaR_max', 'jet_DL1r_max', 'jet_px', 'jet_py',
                 'jet_pz', 'ljet_px', 'ljet_py', 'ljet_pz', 'MET_m', 'MET_eta', 'MET_px',
                 'MET_py','gen_split','train_weight','LABEL']]

s1_test = (s_1.loc[s_1['gen_split']=='test']).drop(columns=['normalisedCombinedWeight','gen_split','train_weight','LABEL'])
s1_train = (s_1.loc[s_1['gen_split']=='train']).drop(columns=['normalisedCombinedWeight','gen_split','train_weight','LABEL'])

y_s1_test = (s_1.loc[s_1['gen_split']=='test'])['LABEL']

nCW_s1 = (s_1.loc[s_1['gen_split']=='test'])['normalisedCombinedWeight']

#

s_2 = s2.loc[:,['normalisedCombinedWeight','MET', 'MET_Phi','jet_pt', 
                 'jet_e', 'jet_eta', 'jet_phi', 'ljet_pt', 'topjet_pt',
                 'ljet_e', 'topjet_e', 'ljet_m', 'topjet_m', 'ljet_eta', 'topjet_eta',
                 'ljet_phi', 'topjet_phi', 'Omega', 'HT', 'Centrality',
                 'DeltaR_max', 'jet_DL1r_max', 'jet_px', 'jet_py',
                 'jet_pz', 'ljet_px', 'ljet_py', 'ljet_pz', 'MET_m', 'MET_eta', 'MET_px',
                 'MET_py','gen_split','train_weight','LABEL']]

s2_test = (s_2.loc[s_2['gen_split']=='test']).drop(columns=['normalisedCombinedWeight','gen_split','train_weight','LABEL'])
s2_train = (s_2.loc[s_2['gen_split']=='train']).drop(columns=['normalisedCombinedWeight','gen_split','train_weight','LABEL'])

y_s2_test = (s_2.loc[s_2['gen_split']=='test'])['LABEL']

nCW_s2 = (s_2.loc[s_2['gen_split']=='test'])['normalisedCombinedWeight']

#

s_3 = s3.loc[:,['normalisedCombinedWeight','MET', 'MET_Phi','jet_pt', 
                 'jet_e', 'jet_eta', 'jet_phi', 'ljet_pt', 'topjet_pt',
                 'ljet_e', 'topjet_e', 'ljet_m', 'topjet_m', 'ljet_eta', 'topjet_eta',
                 'ljet_phi', 'topjet_phi', 'Omega', 'HT', 'Centrality',
                 'DeltaR_max', 'jet_DL1r_max', 'jet_px', 'jet_py',
                 'jet_pz', 'ljet_px', 'ljet_py', 'ljet_pz', 'MET_m', 'MET_eta', 'MET_px',
                 'MET_py','gen_split','train_weight','LABEL']]

s3_test = (s_3.loc[s_3['gen_split']=='test']).drop(columns=['normalisedCombinedWeight','gen_split','train_weight','LABEL'])
s3_train = (s_3.loc[s_3['gen_split']=='train']).drop(columns=['normalisedCombinedWeight','gen_split','train_weight','LABEL'])

y_s3_test = (s_3.loc[s_3['gen_split']=='test'])['LABEL']

nCW_s3 = (s_3.loc[s_3['gen_split']=='test'])['normalisedCombinedWeight']

#

s_4 = s4.loc[:,['normalisedCombinedWeight','MET', 'MET_Phi','jet_pt', 
                 'jet_e', 'jet_eta', 'jet_phi', 'ljet_pt', 'topjet_pt',
                 'ljet_e', 'topjet_e', 'ljet_m', 'topjet_m', 'ljet_eta', 'topjet_eta',
                 'ljet_phi', 'topjet_phi', 'Omega', 'HT', 'Centrality',
                 'DeltaR_max', 'jet_DL1r_max', 'jet_px', 'jet_py',
                 'jet_pz', 'ljet_px', 'ljet_py', 'ljet_pz', 'MET_m', 'MET_eta', 'MET_px',
                 'MET_py','gen_split','train_weight','LABEL']]

s4_test = (s_4.loc[s_4['gen_split']=='test']).drop(columns=['normalisedCombinedWeight','gen_split','train_weight','LABEL'])
s4_train = (s_4.loc[s_4['gen_split']=='train']).drop(columns=['normalisedCombinedWeight','gen_split','train_weight','LABEL'])

y_s4_test = (s_4.loc[s_4['gen_split']=='test'])['LABEL']

nCW_s4 = (s_4.loc[s_4['gen_split']=='test'])['normalisedCombinedWeight']


#----------------
# CONCAT DATASETS
#----------------

X_test_1 = pd.concat([s1_test,bg_test],ignore_index=True)
X_train_1 = pd.concat([s1_train,bg_train],ignore_index=True)
y_test_1 = pd.concat([y_s1_test,y_bg_test],ignore_index=True)
nCW_1 = pd.concat([nCW_s1,nCW_bg],ignore_index=True)

X_test_2 = pd.concat([s2_test,bg_test],ignore_index=True)
X_train_2 = pd.concat([s2_train,bg_train],ignore_index=True)
y_test_2 = pd.concat([y_s2_test,y_bg_test],ignore_index=True)
nCW_2 = pd.concat([nCW_s2,nCW_bg],ignore_index=True)

X_test_3 = pd.concat([s3_test,bg_test],ignore_index=True)
X_train_3 = pd.concat([s3_train,bg_train],ignore_index=True)
y_test_3 = pd.concat([y_s3_test,y_bg_test],ignore_index=True)
nCW_3 = pd.concat([nCW_s3,nCW_bg],ignore_index=True)

X_test_4 = pd.concat([s4_test,bg_test],ignore_index=True)
X_train_4 = pd.concat([s4_train,bg_train],ignore_index=True)
y_test_4 = pd.concat([y_s4_test,y_bg_test],ignore_index=True)
nCW_4 = pd.concat([nCW_s4,nCW_bg],ignore_index=True)


#----------------
# STANDARD SCALER
#----------------


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_1 = pd.DataFrame(scaler.fit_transform(X_train_1),columns = X_train_1.columns)
X_test_1 = pd.DataFrame(scaler.transform(X_test_1),columns = X_test_1.columns)
X_val_1 = pd.DataFrame(scaler.transform(X_val_1),columns = X_val_1.columns)

X_train_2 = pd.DataFrame(scaler.fit_transform(X_train_2),columns = X_train_2.columns)
X_test_2 = pd.DataFrame(scaler.transform(X_test_2),columns = X_test_2.columns)
X_val_2 = pd.DataFrame(scaler.transform(X_val_2),columns = X_val_2.columns)

X_train_3 = pd.DataFrame(scaler.fit_transform(X_train_3),columns = X_train_3.columns)
X_test_3 = pd.DataFrame(scaler.transform(X_test_3),columns = X_test_3.columns)
X_val_3 = pd.DataFrame(scaler.transform(X_val_3),columns = X_val_3.columns)

X_train_4 = pd.DataFrame(scaler.fit_transform(X_train_4),columns = X_train_4.columns)
X_test_4 = pd.DataFrame(scaler.transform(X_test_4),columns = X_test_4.columns)
X_val_4 = pd.DataFrame(scaler.transform(X_val_4),columns = X_val_4.columns)


#---------------------
# MODELS' ARCHITECTURE
#---------------------


model_1 = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_1.shape[1],)),
  tf.keras.layers.Dense(1,activation="sigmoid")
])

model_2 = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_2.shape[1],)),
  tf.keras.layers.Dense(1,activation="sigmoid")
])

model_3 = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_3.shape[1],)),
  tf.keras.layers.Dense(1,activation="sigmoid")
])

model_4 = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_4.shape[1],)),
  tf.keras.layers.Dense(1,activation="sigmoid")
])


model_1.compile(loss="binary_crossentropy",
              optimizer="adam",
              weighted_metrics=['accuracy', tf.keras.metrics.AUC(name="auc")])

model_2.compile(loss="binary_crossentropy",
              optimizer="adam",
              weighted_metrics=['accuracy', tf.keras.metrics.AUC(name="auc")])

model_3.compile(loss="binary_crossentropy",
              optimizer="adam",
              weighted_metrics=['accuracy', tf.keras.metrics.AUC(name="auc")])

model_4.compile(loss="binary_crossentropy",
              optimizer="adam",
              weighted_metrics=['accuracy', tf.keras.metrics.AUC(name="auc")])


#---------------
# FITTING MODELS
#---------------


history_1 = model_1.fit(X_train_1, y_train_1.values,
                    epochs=100,
                    validation_data=(X_val_1, y_val_1, W_val_1),
                    batch_size=1024,
                    sample_weight=W_train_1.values,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])

history_2 = model_2.fit(X_train_2, y_train_2.values,
                    epochs=100,
                    validation_data=(X_val_2, y_val_2, W_val_2),
                    batch_size=1024,
                    sample_weight=W_train_2.values,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])

history_3 = model_3.fit(X_train_3, y_train_3.values,
                    epochs=100,
                    validation_data=(X_val_3, y_val_3, W_val_3),
                    batch_size=1024,
                    sample_weight=W_train_3.values,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])

history_4 = model_4.fit(X_train_4, y_train_4.values,
                    epochs=100,
                    validation_data=(X_val_4, y_val_4, W_val_4),
                    batch_size=1024,
                    sample_weight=W_train_4.values,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])


#-------------------------------------
# model_1 PREDICTIONS ON OTHER SIGNALS
#-------------------------------------


preds1_1 = model_1.predict(X_test_1)
preds1_2 = model_1.predict(X_test_2)
preds1_3 = model_1.predict(X_test_3)
preds1_4 = model_1.predict(X_test_4)


#------------------------------------------
# MODELS' PREDICTIONS ON RESPECTIVE SIGNALS
#------------------------------------------

preds1 = model_1.predict(X_test_1)
preds2 = model_2.predict(X_test_2)
preds3 = model_3.predict(X_test_3)
preds4 = model_4.predict(X_test_4)


import csv

with open('NN_1.txt', 'w') as f:
    csv.writer(f, delimiter=' ').writerows(preds1)

with open('NN_2.txt', 'w') as f:
    csv.writer(f, delimiter=' ').writerows(preds2)

with open('NN_3.txt', 'w') as f:
    csv.writer(f, delimiter=' ').writerows(preds3)

with open('NN_4.txt', 'w') as f:
    csv.writer(f, delimiter=' ').writerows(preds4)


#----------------------------------------
# MODEL SCORE OF model_1 ON OTHER SIGNALS
#----------------------------------------

def hist_log(preds1_1,preds1_2,preds1_3,preds1_4,y_test_1,y_test_2,y_test_3,y_test_4,nCW_1,nCW_2,nCW_3,nCW_4):
    plt.figure()
    plt.hist(preds1_1[y_test_1 == 0],
           color='b', alpha=0.5, 
           bins=100,
           histtype='stepfilled', density=True,
           label='Background', weights=nCW_1[y_test_1 == 0])

    plt.hist(preds1_1[y_test_1 == 1],
           color='r', alpha=0.5,
           bins=100,
           histtype='step', density=True,
           label='ResmMed4000mX1lb0p2yp0p4', weights=nCW_1[y_test_1 == 1])
    
    plt.hist(preds1_2[y_test_2 == 1],
           color='g', alpha=0.5,
           bins=100,
           histtype='step', density=True,
           label='bbA2000_yb2_Zhvvbb', weights=nCW_2[y_test_2 == 1])
        
    plt.hist(preds1_3[y_test_3 == 1],
           color='y', alpha=0.5,
           bins=100,
           histtype='step', density=True,
           label='GG_direct_2000_0', weights=nCW_3[y_test_3 == 1])
    
    plt.hist(preds1_4[y_test_4 == 1],
           color='k', alpha=0.5,
           bins=100,
           histtype='step', density=True,
           label='HVT_Agv1_VzZH_vvqq_m1000', weights=nCW_4[y_test_4 == 1])
    
    plt.legend()
    plt.title("NN model 1 score on every signal")
    plt.xlabel('Anomaly Score')
    plt.ylabel('Occurrences')
    plt.yscale('log')
    plt.xlim([0,1.01])
    plt.show()
    
hist_log(preds1_1,preds1_2,preds1_3,preds1_4,y_test_1,y_test_2,y_test_3,y_test_4,nCW_1,nCW_2,nCW_3,nCW_4)


#-----------------------------------------
# MODEL SCORE FOR EACH SIGNAL RESPECTIVELY
#-----------------------------------------

def hist(pred,test,weight):
    plt.figure()
    plt.hist(pred[test == 0],
           color='b', alpha=0.5, 
           bins=100,
           histtype='stepfilled', density=True,
           label='Background', weights=weight[test == 0])
    plt.hist(pred[test == 1],
           color='r', alpha=0.5,
           bins=100,
           histtype='stepfilled', density=True,
          label='ResmMed4000mX1lb0p2yp0p4', weights=weight[test == 1])
    plt.legend()
    plt.title("NN model 1 score")
    plt.xlabel('Anomaly Score')
    plt.ylabel('Occurrences')
    plt.yscale('log')
    plt.xlim([0,1])
    plt.show()

hist(preds1,y_test_1,nCW_1)


def hist(pred,test,weight):
    plt.figure()
    plt.hist(pred[test == 0],
           color='b', alpha=0.5, 
           bins=100,
           histtype='stepfilled', density=True,
           label='Background', weights=weight[test == 0])
    plt.hist(pred[test == 1],
           color='g', alpha=0.5,
           bins=100,
           histtype='stepfilled', density=True,
          label='bbA2000_yb2_Zhvvbb', weights=weight[test == 1])
    plt.legend()
    plt.title("NN model 2 score")
    plt.xlabel('Anomaly Score')
    plt.ylabel('Occurrences')
    plt.yscale('log')
    plt.xlim([0,1.01])
    plt.show()

hist(preds2,y_test_2,nCW_2)


def hist(pred,test,weight):
    plt.figure()
    plt.hist(pred[test == 0],
           color='b', alpha=0.5, 
           bins=100,
           histtype='stepfilled', density=True,
           label='Background', weights=weight[test == 0])
    plt.hist(pred[test == 1],
           color='y', alpha=0.5,
           bins=100,
           histtype='stepfilled', density=True,
          label='GG_direct_2000_0', weights=weight[test == 1])
    plt.legend()
    plt.title("NN model 3 score")
    plt.xlabel('Anomaly Score')
    plt.ylabel('Occurrences')
    plt.yscale('log')
    plt.xlim([0,1.01])
    plt.show()

hist(preds3,y_test_3,nCW_3)


def hist(pred,test,weight):
    plt.figure()
    plt.hist(pred[test == 0],
           color='b', alpha=0.5, 
           bins=100,
           histtype='stepfilled', density=True,
           label='Background', weights=weight[test == 0])
    plt.hist(pred[test == 1],
           color='k', alpha=0.5,
           bins=100,
           histtype='stepfilled', density=True,
          label='HVT_Agv1_VzZH_vvqq_m1000', weights=weight[test == 1])
    plt.legend()
    plt.title("NN model 4 score")
    plt.xlabel('Anomaly Score')
    plt.ylabel('Occurrences')
    plt.yscale('log')
    plt.xlim([0,1.01])
    plt.show()

hist(preds4,y_test_4,nCW_4)


#-------------------------------------
# ROC CURVE FOR model_1 ON EACH SIGNAL
#-------------------------------------

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
    plt.title('ROC curve - NN model 1 for each signal')
    plt.legend()
    plt.xlabel('False Positives Rate')
    plt.ylabel('True Positives Rate')
    plt.show()

from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay

my_plot_roc_curve(preds1_1,y_test_1,preds1_2,y_test_2,preds1_3,y_test_3,preds1_4,y_test_4)


#-------------------------------------
# ROC CURVE FOR EACH RESPECTIVE SIGNAL
#-------------------------------------

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
    plt.title('ROC curve - NN model for each signal')
    plt.legend()
    plt.xlabel('False Positives Rate')
    plt.ylabel('True Positives Rate')
    plt.show()

from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay

my_plot_roc_curve(preds1,y_test_1,preds2,y_test_2,preds3,y_test_3,preds4,y_test_4)