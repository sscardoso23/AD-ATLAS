import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

# IMPORT DATASETS

BG = pd.read_csv(r'/Users/cadodo/Desktop/main/LIP/CSV/bkg.csv')
signals = pd.read_csv(r'/Users/cadodo/Desktop/main/LIP/CSV/ResmMed4000mX1lb0p2yp0p4.csv')

# ADD LABELS

BG['LABEL'] = 0
signals['LABEL'] = 1

# DEFINE TRAIN AND TEST SAMPLES

# BG
X_bg = BG.loc[:,['MET_px', 'MET_py','jet_e', 'jet_px', 'jet_py', 'jet_pz',
                  'ljet_e', 'ljet_px', 'ljet_py','ljet_pz','HT',
                  'jet_DL1r_max','gen_split','train_weight','LABEL']]   # Selecting important features

bg_train = (X_bg.loc[X_bg['gen_split'] == 'train']).drop(columns=['gen_split','train_weight','LABEL'])
bg_test = (X_bg.loc[X_bg['gen_split'] == 'test']).drop(columns=['gen_split','train_weight','LABEL'])

y_bg = BG.loc[:,['gen_split','LABEL']]  # Selecting the LABEL

y_bg_test = (y_bg.loc[y_bg['gen_split'] == 'test']).drop(columns=['gen_split'])

# SIGNALS
X_signals = signals.loc[:,['MET_px', 'MET_py','jet_e', 'jet_px', 'jet_py', 'jet_pz',
                  'ljet_e', 'ljet_px', 'ljet_py','ljet_pz','HT',
                  'jet_DL1r_max','gen_split','train_weight','LABEL']]  # Selecting important features

signals_test = (X_signals.loc[X_signals['gen_split'] == 'test']).drop(columns=['gen_split','train_weight','LABEL'])

print(len(signals_test))

y_signals = signals.loc[:,['gen_split','LABEL']]   # Selecting the LABEL

y_signals_test = (y_signals.loc[y_signals['gen_split'] == 'test']).drop(columns=['gen_split'])


# DEFINE WEIGHTS

weight_train_bg = (X_bg.loc[X_bg['gen_split'] == 'train'])[['train_weight','LABEL']]   
weight_test_bg = (X_bg.loc[X_bg['gen_split'] == 'test'])[['train_weight','LABEL']]

weight_train_signals = (X_signals.loc[X_signals['gen_split'] == 'train'])[['train_weight','LABEL']]
weight_test_signals = (X_signals.loc[X_signals['gen_split'] == 'test'])[['train_weight','LABEL']]


# CONCAT DATASETS

X_test = pd.concat([signals_test, bg_test], ignore_index=True) 
y_test = pd.concat([y_signals_test, y_bg_test], ignore_index=True)
weight_test = pd.concat([weight_test_signals, weight_test_bg], ignore_index=True)

y_test = y_test.values
weight_test_ = weight_test['train_weight'].values.reshape(-1,1)


# CALCULATE CONTAMINATION (argument for IF)

contamination = signals['normalisedCombinedWeight'].sum()*100/(signals['normalisedCombinedWeight'].sum()+
                                                              BG['normalisedCombinedWeight'].sum())

# MODEL TRAIN (BG ONLY)

from sklearn.ensemble import IsolationForest

model_IF = IsolationForest(n_estimators = 100, contamination=contamination*0.01, random_state = 0)
model_IF.fit(bg_train)

# MODEL PREDICTIONS

y_pred_model = model_IF.predict(X_test).reshape(-1)  # -1 or 1
y_pred_scores = model_IF.score_samples(X_test).reshape(-1)  # [-1,0]

# PREDICTIONS NORMALISATION  ([-1,0] -> [0,1])

for i in range(len(y_pred_scores)):
    y_pred_scores[i] = -y_pred_scores[i]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
y_pred_scores = y_pred_scores.reshape(-1, 1)
y_pred_scores = scaler.fit_transform(y_pred_scores)

# PLOT MODEL SCORE

density=True

plt.hist(y_pred_scores[y_test == 0],
         color='b', alpha=0.5, 
         bins=30,
         histtype='stepfilled', density=density,
         label='BG (test)', weights=weight_test_[y_test == 0])
plt.hist(y_pred_scores[y_test == 1],
         color='r', alpha=0.5,
         bins=30,
         histtype='stepfilled', density=density,
         label='signals (test)', weights=weight_test_[y_test == 1])
plt.legend()
plt.title("Model score - Isolation Forest (bg_train only)")
plt.show()

# PLOT ROC CURVE

def my_plot_roc_curve(model, y_scores, y_test):
        
    if len(y_scores.shape) == 2:
        if y_scores.shape[1] == 1:
            y_scores = y_scores.reshape(-1)
        elif y_scores.shape[1] == 2:
            y_scores = y_scores[:,1].reshape(-1)

    fpr, tpr, _ = roc_curve(y_test, -y_scores, pos_label=0)
    roc_auc = roc_auc_score(y_test, y_scores)
    plt.clf()
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=model.__class__.__name__)
    display.plot()
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.title('ROC curve - IsolationForest')
    plt.show()

from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay 
my_plot_roc_curve(model_IF, y_pred_scores, y_test)
