# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 16:42:48 2023

@author: xincao
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, InputLayer, Dense, Conv2D, Flatten, BatchNormalization, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import scipy.stats as st
from scipy.stats import gaussian_kde
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy import stats
from scipy.stats import bootstrap
print(tf.keras.backend.image_data_format())
import os
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D,MaxPool1D, MaxPooling1D, AveragePooling1D, MaxPool2D, GlobalAveragePooling1D, GlobalMaxPool1D, Input, Dense, Dropout, Embedding, LSTM, Bidirectional, Conv1D, BatchNormalization, concatenate, Activation, Multiply, Permute, Reshape, Lambda, Add,multiply,Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomUniform,glorot_normal,VarianceScaling
from tensorflow.keras.layers import Attention
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import Callback
import time

df= pd.read_csv('crisprSQL.csv',low_memory=False,na_values=['NA', '', ' '])
dna = list(df['target_sequence'].to_numpy())
grna = list(df['grna_target_sequence'].to_numpy())
label = list(df['log_label'].to_numpy())

def encoding(target_seq, grna_seq):
    
    dict1 = {'A': [1, 0, 0, 0, 0],'T': [0, 1, 0, 0, 0], 
             'G': [0, 0, 1, 0, 0],'C': [0, 0, 0, 1, 0],
             '-': [0, 0, 0, 0, 1]}
    dict2 = {'A':5, 'G':4, 'C':3, 'T':2, '-':1}
     
    def grna_encode(grna_single_seq):
        code_list = []
        for i in range(len(grna_single_seq)):
            code_list.append(dict1[grna_single_seq[i]])
        return code_list
    
    def dna_encode(target_single_seq):
        code_list = []
        for i in range(len(target_single_seq)):
            code_list.append(dict1[target_single_seq[i]])
        return code_list

    or_code = []
    dir_code = np.zeros(2)
    single_seq = []
    result1 = []
    result2 = []
    
    for ii in range(len(target_seq)):
        for jj in range(len(target_seq[ii])):
            temp1 = (np.bitwise_or(dna_encode(target_seq[ii][jj]), grna_encode(grna_seq[ii][jj]))).squeeze()   
            if dict2[target_seq[ii][jj]] == dict2[grna_seq[ii][jj]]:
                pass
            else:
                if dict2[target_seq[ii][jj]] > dict2[grna_seq[ii][jj]]:
                    dir_code[0] = 1
                else:
                    dir_code[1] = 1
            temp2 = np.concatenate((temp1,dir_code))       
            temp2 = np.array(temp2)
            single_seq.append(temp2)
            
            if jj == len(target_seq[ii])-1:
                single_seq = np.array(single_seq)
                result1.append(single_seq)
                single_seq = []   
                
    result2 = np.array(result1)
    
    return result2 


X = encoding(dna,grna)
Y = np.array(label).reshape(-1,1)
X = np.asarray(X).astype('float32')
Y = np.asarray(Y).astype('float32')
xt = X.transpose(0,2,1)
del xt, dna, grna, label

inputRow = 7
inputCol = 23
X=X
Y=Y
indices = np.arange(Y.shape[0])

train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

df_train = df.loc[train_indices]
df_test = df.loc[test_indices]
xtrain = X[train_indices]
ytrain = Y[train_indices]
xtest = X[test_indices]
ytest = Y[test_indices]

def transformImages(
        xtrain, xtest,
        ytrain, ytest,
        imgrows, imgcols,
                    ):
    if tf.keras.backend.image_data_format() == 'channels_first':
        xtrain = xtrain.reshape(xtrain.shape[0], imgrows * imgcols)
        xtest = xtest.reshape(xtest.shape[0], imgrows * imgcols)
        input_shape = (1, imgrows, imgcols)
    else:
        xtrain = xtrain.reshape(xtrain.shape[0], imgrows * imgcols)
        xtest = xtest.reshape(xtest.shape[0], imgrows * imgcols)
        input_shape = (imgrows * imgcols)
    xtrain = xtrain.astype('float32')
    xtest = xtest.astype('float32')
    return xtrain, xtest, ytrain, ytest, input_shape
xtrain, xtest, ytrain, ytest, input_shape = transformImages(xtrain,xtest,ytrain,ytest,inputRow,inputCol)
X2 = X.transpose(0,2,1)
X = X2.reshape((len(X), inputRow*inputCol))

VOCAB_SIZE = 7
EMBED_SIZE = 90
MAXLEN = 161

def residual_block(x, num_filters):
    res = x
    x = Conv1D(num_filters, kernel_size=5, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(num_filters, kernel_size=5, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Add()([res, x])
    x = Activation('relu')(x) 
    return x

def CRISPR_HW():
    input = Input(shape=(input_shape,)) 
    embedded = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=MAXLEN)(input)
    conv1 = Conv1D(70, 4, activation="relu", name="conv1")(embedded)
    batchnor1 = BatchNormalization()(conv1)
    conv2 = Conv1D(40, 6, activation="relu", name="conv2")(batchnor1)
    batchnor2 = BatchNormalization()(conv2)
    batchnor2 = Dropout(0.0)(batchnor2)

    #---------Resnet---------#
    c1 = residual_block(batchnor2, 40)
    c11 = BatchNormalization()(c1)

    # ---------LSTM---------#
    c22 = Bidirectional(LSTM(20, return_sequences=True, activation='relu'))(batchnor2)
    c22 = BatchNormalization()(c22)

    # ---------Attention---------#
    c31 = Conv1D(40, 9, activation='relu', name="c31")(embedded)
    batchnor3 = BatchNormalization()(c31)
    c32 = Attention()([batchnor2, batchnor3])


    merged = concatenate([c11, c22, c32])
    flat = Flatten()(merged)

    dense1 = Dense(300, activation="relu", name="dense1")(flat)
    drop1 = Dropout(0.2)(dense1)

    dense2 = Dense(150, activation="relu", name="dense2")(drop1)
    drop2 = Dropout(0.2)(dense2)

    output = Dense(1, activation="linear", name="dense3")(drop2)
    model = Model(inputs=[input], outputs=[output])
    return model

loss = []
validation_loss = []
pearson = []
spearman = []
# epoch number (EN); batch size (BS); learning rate (LR)
EN = 3#30 
BS = 200
LR = 0.003 

model = CRISPR_HW()
model.summary()
model.compile(
            optimizer=keras.optimizers.Adam(learning_rate = LR),
            loss = 'mse',
            metrics=['mse']
            )
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5, verbose=1)
hist = model.fit(xtrain, ytrain,
                     batch_size = BS,
                     epochs = EN,
                     validation_data = (xtest, ytest),
                     verbose = 1,
                     callbacks=[reduce_lr]
                     )

loss.append(hist.history['loss'])
validation_loss.append([hist.history['val_loss']])
predictions = model.predict(xtest)
coef, p = spearmanr(np.squeeze(ytest), np.squeeze(predictions))
corr, p1 = pearsonr(np.squeeze(ytest), np.squeeze(predictions))
pearson.append(corr)
spearman.append(coef)

plt.figure()
plt.plot(hist.history['loss'], label='Train Loss')
plt.plot(hist.history['val_loss'], label='Validation Loss')
plt.title('FNN5 Training Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
print("pearson: ", np.squeeze(pearson))
print("spearman: ", np.squeeze(spearman))

# %%
df_test['pred'] = predictions
pred_logback = np.exp(predictions) - 1.0
df_test['pred_logback'] = pred_logback

def select_top_percent(df, column_name='pred'):
    top_n = max(1, int(len(df) * 0.05))
    return df.nlargest(top_n, column_name)
top_percent_df = select_top_percent(df_test)

from typing import Sequence
from sklearn import metrics

def cumulative_true(
    y_true: Sequence[float],
    y_pred: Sequence[float]
) -> np.ndarray:
  df = pd.DataFrame({
      'y_true': y_true,
      'y_pred': y_pred,
  }).sort_values(
      by='y_pred', ascending=False)

  return (df['y_true'].cumsum() / df['y_true'].sum()).values


def gini_from_gain(df: pd.DataFrame) -> pd.DataFrame:
  raw = df.apply(lambda x: 2 * x.sum() / df.shape[0] - 1.)
  normalized = raw / raw[0]
  return pd.DataFrame({
      'raw': raw,
      'normalized': normalized
  })[['raw', 'normalized']]

gain_ground_truth = cumulative_true(top_percent_df['cleavage_freq'], top_percent_df['cleavage_freq'])
gain_model = cumulative_true(top_percent_df['cleavage_freq'], top_percent_df['pred_logback'])
gain = pd.DataFrame({
    'ground_truth': gain_ground_truth,
    'model': gain_model
})
gini = gini_from_gain(gain)
print(gini)

#%%
r2 = r2_score(ytest, predictions)
mae = mean_absolute_error(ytest, predictions)
mape = np.mean(np.abs((ytest - predictions) / ytest)) * 100
rmse = np.sqrt(mean_squared_error(ytest, predictions))
mse = mean_squared_error(ytest, predictions)

print()
print(f"Pearson: {pearson}")
print(f"Spearman: {spearman}")
print()
print(f"R²: {r2:.4f}")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

#%%
import pandas as pd
import numpy as np

def select_top_n_percent(df, column, top_n_percent):
    
    sorted_df = df.sort_values(by=column, ascending=False)
    
    num_elements = len(df)
    top_n_count = max(int(np.ceil((top_n_percent / 100) * num_elements)), 1)
    
    top_df = sorted_df.head(top_n_count)
    
    top_values = top_df[column].values
    top_indices = top_df.index.values  
    
    return top_values, top_indices


top5_label, top5_indices = select_top_n_percent(df_test, "cleavage_freq", 5)
top1_label, top1_indices = select_top_n_percent(df_test, "cleavage_freq", 1)

top1_pred = df_test.loc[top1_indices, 'pred_logback'].values
top1_01label = df_test.loc[top1_indices, '01_label'].values

top5_pred = df_test.loc[top5_indices, 'pred_logback'].values
top5_01label = df_test.loc[top5_indices, '01_label'].values

top5_MAPE = np.mean(np.abs((top5_label - top5_pred) / top5_label)) * 100
top1_MAPE = np.mean(np.abs((top1_label - top1_pred) / top1_label)) * 100

print("Top 5% MAPE:", top5_MAPE)
print("Top 1% MAPE:", top1_MAPE)
