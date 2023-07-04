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
import optuna
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from itertools import combinations


signals = pd.read_csv(r'/home/scardoso/ResmMed4000mX1lb0p2yp0p4.csv')
BG = pd.read_csv(r'/home/scardoso/bkg.csv')

BG['LABEL'] = 0
signals['LABEL'] = 1

print('Read BG & signals - DONE')


#-------------
# SPLITTING BG
#-------------

# INPUTS
X_bg = BG.loc[:,['normalisedCombinedWeight','MET_px', 'MET_py','jet_e', 'jet_px', 'jet_py', 'jet_pz',
                  'ljet_e', 'ljet_px', 'ljet_py','ljet_pz','HT','jet_DL1r_max',
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

print('Splitting BG - DONE')


#------------------
# SPLITTING SIGNALS
#------------------

# INPUTS
X_signals = signals.loc[:,['normalisedCombinedWeight','MET_px', 'MET_py','jet_e', 'jet_px', 'jet_py', 'jet_pz',
                  'ljet_e', 'ljet_px', 'ljet_py','ljet_pz','HT','jet_DL1r_max',
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

print('Splitting signals - DONE')


#---------------
# DEFINE WEIGHTS
#---------------

# BG WEIGHTS
weight_bg_train = (X_bg.loc[X_bg['gen_split'] == 'train'])['train_weight'] 
weight_bg_test = (X_bg.loc[X_bg['gen_split'] == 'test'])['train_weight']
weight_bg_val = (X_bg.loc[X_bg['gen_split'] == 'val'])['train_weight']

# SIGNALS WEIGHTS
weight_signals_train = (X_signals.loc[X_signals['gen_split'] == 'train'])['train_weight']
weight_signals_test = (X_signals.loc[X_signals['gen_split'] == 'test'])['train_weight']
weight_signals_val = (X_signals.loc[X_signals['gen_split'] == 'val'])['train_weight']

print('Weights - DONE')


#--------------
# CLASS WEIGHTS
#--------------

class_weights_train = (weight_bg_train.values.sum(),weight_signals_train.values.sum())
class_weights_test = (weight_bg_test.values.sum(),weight_signals_test.values.sum())
class_weights_val = (weight_bg_val.values.sum(),weight_signals_val.values.sum())

print("class_weights_train (BG, signals):",class_weights_train)
print("class_weights_test (BG, signals):",class_weights_test)
print("class_weights_val (BG, signals):",class_weights_val)

print('Class Weights - DONE')


#----------
# JOIN CSVs
#----------

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
nCW_test = pd.concat([nCW_signals_test, nCW_bg_test], ignore_index=True)

print('Join CSVs - DONE')


#--------------------------
# STANDARDISATION OF INPUTS
#--------------------------

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


#-------
# OPTUNA
#-------

from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout

def start_Autoencoder(X_train,X_val,weight_train,weight_val,trials,plot_graph = False):
    # Convert non-pandas array to numpy
    if isinstance(X_train, pd.core.frame.DataFrame) == True:
        X_train = X_train.to_numpy()
    if isinstance(X_val, pd.core.frame.DataFrame) == True:
        X_val = X_val.to_numpy()
    if isinstance(weight_train, pd.core.frame.DataFrame) == True:
        weight_train = weight_train.to_numpy()
    if isinstance(weight_val, pd.core.frame.DataFrame) == True:
        weight_val = weight_val.to_numpy()
        
    input_dim = X_train.shape[1]

#----------------------
# AUTOENCODER STRUCTURE
#----------------------

    def create_model(activation, layers_num, dropout_prob, kernel_initializer):
        K.clear_session()
        autoencoder = Sequential()
    
        # Encoder
        for i in range(layers_num, 1, -1):
            autoencoder.add(Dense(int((input_dim/(2**2))*i**3), 
                            input_shape=(input_dim,), 
                            activation=activation,
                            kernel_initializer=kernel_initializer, 
                            name='encoder' + str(layers_num-i)))
            if (i == layers_num):
                autoencoder.add(Dropout(dropout_prob))

        # Bottleneck  CHANGE HERE THE BOTTLENECK
        autoencoder.add(Dense(2, 
                        activation=activation,
                        kernel_initializer=kernel_initializer, 
                        name='bottleneck' + str(layers_num)))
        # Decoder
        for i in range(2, layers_num+1):
            autoencoder.add(Dense(int((input_dim/(2**2))*i**3), 
                          activation=activation,
                          kernel_initializer=kernel_initializer, 
                          name='decoder' + str(i+layers_num)))
            if (i == layers_num-1):
                autoencoder.add(Dropout(dropout_prob))

        # Output layer
        autoencoder.add(Dense(input_dim, 
                          activation=activation,
                          kernel_initializer=kernel_initializer, 
                          name='output'))
        return autoencoder

#-------------------
# OBJECTIVE FUNCTION
#-------------------

    def objective(trial):
        activation = trial.suggest_categorical("activation", ["relu", "sigmoid", "swish"])
        layers_num = trial.suggest_int("layers_num", 2,5,1)
        dropout_rate = trial.suggest_float("dropout_prob", 0.0, 0.9, step=0.1)
        
        if (activation == "relu"):
            model = create_model(activation, layers_num, dropout_rate, kernel_initializer="HeUniform")
        else:
            model = create_model(activation, layers_num, dropout_rate, kernel_initializer="GlorotUniform")
            
        model.compile(optimizer='adam', loss='mse', weighted_metrics=[tf.keras.metrics.MeanSquaredError()])
        
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10,  restore_best_weights=True)
        
        history = model.fit(X_train, X_train,
                        batch_size = 1024,
                        epochs = 100,
                        validation_data = (X_val, X_val, weight_val),
                        sample_weight = weight_train,
                        callbacks = [callback], 
                        verbose = 0)
        return history.history["loss"][-1]

#--------------
# STUDY SESSION
#--------------

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=trials)

#------------
# FINAL MODEL
#------------

    print('Best hyperparams found by Optuna: \n', study.best_params)
    if (study.best_params['activation'] == "relu"):
        model = create_model(study.best_params['activation'],
                         int(study.best_params['layers_num']),
                         study.best_params['dropout_prob'],
                         kernel_initializer = "HeUniform")
    else:
        model = create_model(study.best_params['activation'],
                        int(study.best_params['layers_num']),
                        study.best_params['dropout_prob'],
                        kernel_initializer = "GlorotUniform")

  
    model.compile(optimizer = 'adam', loss = 'mse', weighted_metrics=[tf.keras.metrics.MeanSquaredError()])
    model.summary()
    # Implement early stopping criterion. 
    # Training process stops when there is no improvement during 50 iterations
    callback = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, X_train,
                        batch_size = 1024,
                        epochs = 100,
                        validation_data = (X_val, X_val, weight_val),
                        sample_weight = weight_train,
                        callbacks = [callback], 
                        verbose = 0)
    result = model.predict(X_train)
    
    print('Model predicts: \n', result)

#-------------
# PLOT RESULTS
#-------------

    plot_model(model, 
            to_file='model_plot.png', 
            show_shapes=True, 
            show_layer_names=True,
            rankdir="TB",
            dpi=150)
    
    if (plot_graph == True):
        pd.DataFrame(history.history).plot(figsize=(8,5))
        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel('MSE')
        plt.title('Loss curve')
        plt.savefig('loss_curve.png', dpi=150)
    
#------------------
# RESULT EVALUATION
#------------------

    # Result evaluation
    print(f'RMSE Autoencoder: {np.sqrt(mean_squared_error(X_train, result))}')
    print('')
    feature_extractor = keras.Model(
        inputs = model.inputs,
        outputs = model.get_layer(name = 'bottleneck' + str(study.best_params['layers_num'])).output)
    # Following values are returned: extracted_f || MSE || OPTUNA best hyperparams
    return np.array(feature_extractor(X_train)), mean_squared_error(X_train, result), study.best_params

Acoder_representation, Acoder_MSE, Acoder_hyperparams = start_Autoencoder(X_train,X_val,weight_train,weight_val,
                                                                          trials = 10, plot_graph=True)
