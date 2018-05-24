# Visual unimodal multitask learning (valence+polarity+intensity)
# for ACL2018 Computational Modeling of Human Multimodal Language Workshop paper

from __future__ import print_function
import numpy as np
import pandas as pd
from collections import defaultdict
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Merge, Input, Flatten
from keras.optimizers import RMSprop,Adamax
from keras.callbacks import EarlyStopping
from keras.regularizers import l1, l2
from keras import backend as K
from mmdata import MOSI, Dataset

# turn off the warnings, be careful when use this
import warnings
warnings.filterwarnings("ignore")

# save outputs to a log file in case there is a broken pipe
import sys
idlestdout = sys.stdout
logger = open("prediction/output_V_unimodal_tri.txt", "w")
sys.stdout = logger

# custom evaluation metrics
def pearson_cc(y_true, y_pred):
    fsp = y_pred - K.mean(y_pred,axis=0)   
    fst = y_true - K.mean(y_true,axis=0) 
    devP = K.std(y_pred,axis=0)  
    devT = K.std(y_true,axis=0)

    return K.sum(K.mean(fsp*fst,axis=0)/(devP*devT))

def pad(data, max_len):
    """A funtion for padding/truncating sequence data to a given lenght"""
    # recall that data at each time step is a tuple (start_time, end_time, feature_vector), we only take the vector
    data = np.array([feature[2] for feature in data])
    n_rows = data.shape[0]
    dim = data.shape[1]
    if max_len >= n_rows:
        diff = max_len - n_rows
        padding = np.zeros((diff, dim))
        padded = np.concatenate((padding, data))
        return padded
    else:
        return data[-max_len:]

# meta parameters
maxlen = 15 # Each utterance will be truncated/padded to 15 words
batch_size = 128
nb_epoch = 1000 # number of total epochs to train the model
# if the validation loss isn't decreasing for a number of epochs, stop training to prevent over-fitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

opt_func = Adamax(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08) # optimization function
loss_func_main = 'mae' # loss function
metr_main = 'mae' # evaluation metric
weight_main = 1.0 # weight for multitask learning
# for Valence polarity classification
loss_func_aux1 = 'binary_crossentropy' # loss function
metr_aux1 = 'binary_accuracy' # evaluation metric
weight_aux1 = 0.5 # weight for multitask learning
# for Valence intensity classification
loss_func_aux2 = 'categorical_crossentropy' # loss function
metr_aux2 = 'accuracy' # evaluation metric
weight_aux2 = 0.5 # weight for multitask learning

# Download the data if not present
mosi = MOSI()
covarep = mosi.covarep() # features
facet = mosi.facet() # features
embeddings = mosi.embeddings() # features
sentiments = mosi.sentiments() # Valence labels
train_ids = mosi.train()
valid_ids = mosi.valid()
test_ids = mosi.test()

bimodal = Dataset.merge(embeddings, facet)
dataset = bimodal.align('embeddings')

# Some data preprocessing
print("Preparing train and test data...")
# sort through all the video ID, segment ID pairs
train_set_ids = []
for vid in train_ids:
    for sid in dataset['embeddings'][vid].keys():
        if dataset['embeddings'][vid][sid] and dataset['facet'][vid][sid]:
            train_set_ids.append((vid, sid))

valid_set_ids = []
for vid in valid_ids:
    for sid in dataset['embeddings'][vid].keys():
        if dataset['embeddings'][vid][sid] and dataset['facet'][vid][sid]:
            valid_set_ids.append((vid, sid))

test_set_ids = []
for vid in test_ids:
    for sid in dataset['embeddings'][vid].keys():
        if dataset['embeddings'][vid][sid] and dataset['facet'][vid][sid]:
           test_set_ids.append((vid, sid))

# partition the training, valid and test set. all sequences will be padded/truncated to 15 steps
# data will have shape (dataset_size, max_len, feature_dim)
max_len = 15

x_train = np.stack([pad(dataset['facet'][vid][sid], max_len) for (vid, sid) in train_set_ids], axis=0)
x_valid = np.stack([pad(dataset['facet'][vid][sid], max_len) for (vid, sid) in valid_set_ids], axis=0)
x_test = np.stack([pad(dataset['facet'][vid][sid], max_len) for (vid, sid) in test_set_ids], axis=0)

# sentiment scores
y_train = np.array([sentiments[vid][sid] for (vid, sid) in train_set_ids])
y_valid = np.array([sentiments[vid][sid] for (vid, sid) in valid_set_ids])
y_test = np.array([sentiments[vid][sid] for (vid, sid) in test_set_ids])

# Binary polarity and intensity classes
z1_train = []
z2_train = []
for (vid, sid) in train_set_ids:
    Polarity_label = 1 if sentiments[vid][sid] >= 0 else 0 # binarize the Valence labels for polarity
    if abs(sentiments[vid][sid]) >= 2.5:
        Intensity_label = [0,0,0,1] # strong
    elif abs(sentiments[vid][sid]) >= 1.5:
        Intensity_label = [0,0,1,0] # medium
    elif abs(sentiments[vid][sid]) >= 0.5:
        Intensity_label = [0,1,0,0] # weak
    else:
        Intensity_label = [1,0,0,0] # neutral
    z1_train.append(Polarity_label)
    z2_train.append(Intensity_label)

z1_valid = []
z2_valid = []
for (vid, sid) in valid_set_ids:
    Polarity_label = 1 if sentiments[vid][sid] >= 0 else 0 # binarize the Valence labels for polarity
    if abs(sentiments[vid][sid]) >= 2.5:
        Intensity_label = [0,0,0,1] # strong
    elif abs(sentiments[vid][sid]) >= 1.5:
        Intensity_label = [0,0,1,0] # medium
    elif abs(sentiments[vid][sid]) >= 0.5:
        Intensity_label = [0,1,0,0] # weak
    else:
        Intensity_label = [1,0,0,0] # neutral
    z1_valid.append(Polarity_label)
    z2_valid.append(Intensity_label)

z1_test = []
z2_test = []
for (vid, sid) in test_set_ids:
    Polarity_label = 1 if sentiments[vid][sid] >= 0 else 0 # binarize the Valence labels for polarity
    if abs(sentiments[vid][sid]) >= 2.5:
        Intensity_label = [0,0,0,1] # strong
    elif abs(sentiments[vid][sid]) >= 1.5:
        Intensity_label = [0,0,1,0] # medium
    elif abs(sentiments[vid][sid]) >= 0.5:
        Intensity_label = [0,1,0,0] # weak
    else:
        Intensity_label = [1,0,0,0] # neutral
    z1_test.append(Polarity_label)
    z2_test.append(Intensity_label)

z1_train = np.asarray(z1_train)
z1_valid = np.asarray(z1_valid)
z1_test = np.asarray(z1_test)
z2_train = np.asarray(z2_train)
z2_valid = np.asarray(z2_valid)
z2_test = np.asarray(z2_test)

# normalize covarep and facet features, remove possible NaN values
visual_max = np.max(np.max(np.abs(x_train), axis=0), axis=0)
visual_max[visual_max==0] = 1 # if the maximum is 0 we don't normalize this dimension
x_train = x_train / visual_max
x_valid = x_valid / visual_max
x_test = x_test / visual_max

x_train[x_train != x_train] = 0
x_valid[x_valid != x_valid] = 0
x_test[x_test != x_test] = 0

print("Data preprocessing finished! Begin compiling and training model.")

# Building model
all_input = Input(shape=(maxlen, 46), dtype='float32', name='input')
h1 = Dropout(0.2)(all_input)
h2 = Dense(32, activation='relu', W_regularizer=l2(0.0), trainable=True)(h1)
h3 = Dense(32, activation='relu', W_regularizer=l2(0.0), trainable=True)(h2)
h4 = Dense(32, activation='relu', W_regularizer=l2(0.0), trainable=True)(h3)
h5 = Flatten()(h4)
main_output = Dense(1, activation='tanh', name='main_output')(h5) # valence regression
auxiliary_output_1 = Dense(1, activation='sigmoid', name='aux_output_1')(h5) # Polarity classification
auxiliary_output_2 = Dense(4, activation='softmax', name='aux_output_2')(h5) # Intensity classification
model = Model(inputs=all_input, outputs=[main_output, auxiliary_output_1, auxiliary_output_2])

# try using different optimizers and different optimizer configs
model.compile(optimizer=opt_func,
              loss={'main_output': loss_func_main, 'aux_output_1': loss_func_aux1, 'aux_output_2': loss_func_aux2},
              loss_weights={'main_output': weight_main, 'aux_output_1': weight_aux1, 'aux_output_2': weight_aux2},
              metrics={'main_output': [pearson_cc,metr_main], 'aux_output_1': metr_aux1, 'aux_output_2': metr_aux2})

print('Training...')
model.fit(x_train,
          {'main_output': y_train, 'aux_output_1': z1_train, 'aux_output_2': z2_train},
          batch_size=batch_size,
          epochs=nb_epoch,
          validation_data=[x_valid, {'main_output': y_valid, 'aux_output_1': z1_valid, 'aux_output_2': z2_valid}],
          callbacks=[early_stopping])

# Evaluation
print('\n\nEvaluating on train set...')
trn_score, trn_score_emo, trn_score_v1, trn_score_v2, trn_cc_emo, trn_mae_emo, trn_mae_v1, trn_mae_v2 = model.evaluate(x_train, {'main_output': y_train, 'aux_output_1': z1_train, 'aux_output_2': z2_train}, batch_size=batch_size)
print('Valence Train cc:', trn_cc_emo)
print('Valence Train mae:', trn_mae_emo)
print('Binary Polarity Train accuracy:', trn_mae_v1)
print('Intensity Train accuracy:', trn_mae_v2)
print('\nEvaluating on valisation set...')
val_score, val_score_emo, val_score_v1, val_score_v2, val_cc_emo, val_mae_emo, val_mae_v1, val_mae_v2 = model.evaluate(x_valid, {'main_output': y_valid, 'aux_output_1': z1_valid, 'aux_output_2': z2_valid}, batch_size=batch_size)
print('Valence Validation cc:', val_cc_emo)
print('Valence Validation mae:', val_mae_emo)
print('Binary Polarity Validation accuracy:', val_mae_v1)
print('Intensity Validation accuracy:', val_mae_v2)
print('\nEvaluating on test set...')
tst_score, tst_score_emo, tst_score_v1, tst_score_v2, tst_cc_emo, tst_mae_emo, tst_mae_v1, tst_mae_v2 = model.evaluate(x_test, {'main_output': y_test, 'aux_output_1': z1_test, 'aux_output_2': z2_test}, batch_size=batch_size)
print('Valence Test cc:', tst_cc_emo)
print('Valence Test mae:', tst_mae_emo)
print('Binary Polarity Test accuracy:', tst_mae_v1)
print('Intensity Test accuracy:', tst_mae_v2)

# output predictions
np.set_printoptions(threshold=np.nan)
tst_pred_file = "prediction/pred_V_unimodal_tri.txt"
print('Printing predictions...')
tst_pred = model.predict(x_test)
tst_df = pd.DataFrame(tst_pred[0])
tst_df.to_csv(tst_pred_file, index=False, header=False)

print('\nDone!')

# Flush outputs to log file
logger.flush()
logger.close()
