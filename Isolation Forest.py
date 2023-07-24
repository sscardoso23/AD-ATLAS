import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt


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

bg_train = (X_bg.loc[X_bg['gen_split'] == 'train']).drop(columns=['normalisedCombinedWeight',
                                                                  'gen_split','train_weight','LABEL'])
bg_test = (X_bg.loc[X_bg['gen_split'] == 'test']).drop(columns=['normalisedCombinedWeight',
                                                                'gen_split','train_weight','LABEL'])

weight_bg = (X_bg.loc[X_bg['gen_split'] == 'train'])['train_weight']

nCW_bg = (X_bg.loc[X_bg['gen_split'] == 'test'])['normalisedCombinedWeight']

y_bg = BG.loc[:,['gen_split','LABEL']]

y_bg_test = (y_bg.loc[y_bg['gen_split'] == 'test']).drop(columns=['gen_split'])

#

X_s1 = s1.loc[:,['normalisedCombinedWeight','MET', 'MET_Phi','jet_pt', 
                           'jet_e', 'jet_eta', 'jet_phi', 'ljet_pt', 'topjet_pt',
                           'ljet_e', 'topjet_e', 'ljet_m', 'topjet_m', 'ljet_eta', 'topjet_eta',
                           'ljet_phi', 'topjet_phi', 'Omega', 'HT', 'Centrality',
                           'DeltaR_max', 'jet_DL1r_max', 'jet_px', 'jet_py',
                           'jet_pz', 'ljet_px', 'ljet_py', 'ljet_pz', 'MET_m', 'MET_eta', 'MET_px',
                           'MET_py','gen_split','train_weight','LABEL']]

s1_test = (X_s1.loc[X_s1['gen_split'] == 'test']).drop(columns=['normalisedCombinedWeight',
                                                                'gen_split','train_weight','LABEL'])

nCW_s1 = (X_s1.loc[X_s1['gen_split'] == 'test'])['normalisedCombinedWeight']

y_s1 = s1.loc[:,['gen_split','LABEL']]

y_s1_test = (y_s1.loc[y_s1['gen_split'] == 'test']).drop(columns=['gen_split'])

#

X_s2 = s2.loc[:,['normalisedCombinedWeight','MET', 'MET_Phi','jet_pt', 
                           'jet_e', 'jet_eta', 'jet_phi', 'ljet_pt', 'topjet_pt',
                           'ljet_e', 'topjet_e', 'ljet_m', 'topjet_m', 'ljet_eta', 'topjet_eta',
                           'ljet_phi', 'topjet_phi', 'Omega', 'HT', 'Centrality',
                           'DeltaR_max', 'jet_DL1r_max', 'jet_px', 'jet_py',
                           'jet_pz', 'ljet_px', 'ljet_py', 'ljet_pz', 'MET_m', 'MET_eta', 'MET_px',
                           'MET_py','gen_split','train_weight','LABEL']]

s2_test = (X_s2.loc[X_s2['gen_split'] == 'test']).drop(columns=['normalisedCombinedWeight',
                                                                'gen_split','train_weight','LABEL'])

nCW_s2 = (X_s2.loc[X_s2['gen_split'] == 'test'])['normalisedCombinedWeight']

y_s2 = s2.loc[:,['gen_split','LABEL']]

y_s2_test = (y_s2.loc[y_s2['gen_split'] == 'test']).drop(columns=['gen_split'])

#

X_s3 = s3.loc[:,['normalisedCombinedWeight','MET', 'MET_Phi','jet_pt', 
                           'jet_e', 'jet_eta', 'jet_phi', 'ljet_pt', 'topjet_pt',
                           'ljet_e', 'topjet_e', 'ljet_m', 'topjet_m', 'ljet_eta', 'topjet_eta',
                           'ljet_phi', 'topjet_phi', 'Omega', 'HT', 'Centrality',
                           'DeltaR_max', 'jet_DL1r_max', 'jet_px', 'jet_py',
                           'jet_pz', 'ljet_px', 'ljet_py', 'ljet_pz', 'MET_m', 'MET_eta', 'MET_px',
                           'MET_py','gen_split','train_weight','LABEL']]

s3_test = (X_s3.loc[X_s3['gen_split'] == 'test']).drop(columns=['normalisedCombinedWeight',
                                                                'gen_split','train_weight','LABEL'])

nCW_s3 = (X_s3.loc[X_s3['gen_split'] == 'test'])['normalisedCombinedWeight']

y_s3 = s3.loc[:,['gen_split','LABEL']]

y_s3_test = (y_s3.loc[y_s3['gen_split'] == 'test']).drop(columns=['gen_split'])

#

X_s4 = s4.loc[:,['normalisedCombinedWeight','MET', 'MET_Phi','jet_pt', 
                           'jet_e', 'jet_eta', 'jet_phi', 'ljet_pt', 'topjet_pt',
                           'ljet_e', 'topjet_e', 'ljet_m', 'topjet_m', 'ljet_eta', 'topjet_eta',
                           'ljet_phi', 'topjet_phi', 'Omega', 'HT', 'Centrality',
                           'DeltaR_max', 'jet_DL1r_max', 'jet_px', 'jet_py',
                           'jet_pz', 'ljet_px', 'ljet_py', 'ljet_pz', 'MET_m', 'MET_eta', 'MET_px',
                           'MET_py','gen_split','train_weight','LABEL']]

s4_test = (X_s4.loc[X_s4['gen_split'] == 'test']).drop(columns=['normalisedCombinedWeight',
                                                                'gen_split','train_weight','LABEL'])

nCW_s4 = (X_s4.loc[X_s4['gen_split'] == 'test'])['normalisedCombinedWeight']

y_s4 = s4.loc[:,['gen_split','LABEL']]

y_s4_test = (y_s4.loc[y_s4['gen_split'] == 'test']).drop(columns=['gen_split'])


#----------------
# CONCAT DATASETS
#----------------


X_test_1 = pd.concat([s1_test, bg_test], ignore_index=True)
X_test_2 = pd.concat([s2_test, bg_test], ignore_index=True) 
X_test_3 = pd.concat([s3_test, bg_test], ignore_index=True) 
X_test_4 = pd.concat([s4_test, bg_test], ignore_index=True) 

y_1 = pd.concat([y_s1_test, y_bg_test], ignore_index=True)
y_2 = pd.concat([y_s2_test, y_bg_test], ignore_index=True)
y_3 = pd.concat([y_s3_test, y_bg_test], ignore_index=True)
y_4 = pd.concat([y_s4_test, y_bg_test], ignore_index=True)


#----------------
# STANDARD SCALER
#----------------


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

bg_train = pd.DataFrame(scaler.fit_transform(bg_train),columns = bg_train.columns)
X_test_1 = pd.DataFrame(scaler.transform(X_test_1),columns = X_test_1.columns)
X_test_2 = pd.DataFrame(scaler.transform(X_test_2),columns = X_test_3.columns)
X_test_3 = pd.DataFrame(scaler.transform(X_test_3),columns = X_test_3.columns)
X_test_4 = pd.DataFrame(scaler.transform(X_test_4),columns = X_test_4.columns)


#----------
# MODEL FIT
#----------


contamination = s1['normalisedCombinedWeight'].sum()*100/(s1['normalisedCombinedWeight'].sum()+
                                                              BG['normalisedCombinedWeight'].sum())


from sklearn.ensemble import IsolationForest

model_IF = IsolationForest(n_estimators = 100, contamination=contamination*0.01, random_state = 0)
model_IF.fit(bg_train, weight_bg)


#---------------
# MODEL PREDICTS
#---------------


preds1 = model_IF.score_samples(X_test_1).reshape(-1)
preds2 = model_IF.score_samples(X_test_2).reshape(-1)
preds3 = model_IF.score_samples(X_test_3).reshape(-1)
preds4 = model_IF.score_samples(X_test_4).reshape(-1)


#----------------------
# SCALE -> [0,1]
#----------------------

for i in range(len(preds1)):
    preds1[i] = -preds1[i]
    
for j in range(len(preds2)):
    preds2[j] = -preds2[j]

for k in range(len(preds3)):
    preds3[k] = -preds3[k]

for m in range(len(preds4)):
    preds4[m] = -preds4[m]
    

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
preds1 = preds1.reshape(-1, 1)
preds1 = scaler.fit_transform(preds1)

preds2 = preds2.reshape(-1, 1)
preds2 = scaler.fit_transform(preds2)

preds3 = preds3.reshape(-1, 1)
preds3 = scaler.fit_transform(preds3)

preds4 = preds4.reshape(-1, 1)
preds4 = scaler.fit_transform(preds4)


import csv

with open('IF1 - ResmMed4000mX1lb0p2yp0p4.txt', 'w') as f:
    csv.writer(f, delimiter=' ').writerows(preds1)

with open('IF1 - bbA2000_yb2_Zhvvbb.txt', 'w') as f:
    csv.writer(f, delimiter=' ').writerows(preds2)

with open('IF1 - GG_direct_2000_0.txt', 'w') as f:
    csv.writer(f, delimiter=' ').writerows(preds3)

with open('IF1 - HVT_Agv1_VzZH_vvqq_m1000.txt', 'w') as f:
    csv.writer(f, delimiter=' ').writerows(preds4)


#------------
# MODEL SCORE
#------------


plt.hist(preds1[y_1 == 0],
         color='b', alpha=0.5, 
         bins=100,
         histtype='stepfilled', density=density,
         label='Background', weights=nCW_bg)
plt.hist(preds1[y_1 == 1],
         color='r', alpha=0.5,
         bins=100,
         histtype='step', density=density,
         label='ResmMed4000mX1lb0p2yp0p4', weights=nCW_s1)
plt.hist(preds2[y_2 == 1],
         color='g', alpha=0.5,
         bins=100,
         histtype='step', density=density,
         label='bbA2000_yb2_Zhvvbb', weights=np.absolute(nCW_s2))
plt.hist(preds3[y_3 == 1],
         color='y', alpha=0.5,
         bins=100,
         histtype='step', density=density,
         label='GG_direct_2000_0', weights=nCW_s3)
plt.hist(preds4[y_4 == 1],
         color='k', alpha=0.5,
         bins=100,
         histtype='step', density=density,
         label='HVT_Agv1_VzZH_vvqq_m1000', weights=nCW_s4)

plt.legend()
plt.xlabel('Anomaly Score')
plt.ylabel('Occurrences')
plt.title("Model score - Isolation Forest 1")


#----------
# ROC CURVE
#----------


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
    plt.title('ROC curve - Isolation Forest 1')
    plt.legend()

from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay

my_plot_roc_curve(preds1,y_1,preds2,y_2,preds3,y_3,preds4,y_4)