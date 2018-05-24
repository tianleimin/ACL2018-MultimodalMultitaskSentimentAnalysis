# Early Fusion multimodal singletask learning
# for ACL2018 Computational Modeling of Human Multimodal Language Workshop paper

from __future__ import print_function
import numpy as np
import pandas as pd
from collections import defaultdict
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Merge, Input, BatchNormalization, Flatten, Reshape
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
logger = open("prediction/output_FL_uno.txt", "w")
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
loss_func = 'mae' # loss function
metr = 'mae' # evaluation metric

# Download the data if not present
mosi = MOSI()
covarep = mosi.covarep() # features
facet = mosi.facet() # features
embeddings = mosi.embeddings() # features
sentiments = mosi.sentiments() # Valence labels
train_ids = mosi.train()
valid_ids = mosi.valid()
test_ids = mosi.test()

# Merge different features and do word level feature alignment (align according to timestamps of embeddings)
bimodal = Dataset.merge(embeddings, facet)
trimodal = Dataset.merge(bimodal, covarep)
dataset = trimodal.align('embeddings')

# Some data preprocessing
print("Preparing train and test data...")
# sort through all the video ID, segment ID pairs
train_set_ids = []
for vid in train_ids:
    for sid in dataset['embeddings'][vid].keys():
        if dataset['embeddings'][vid][sid] and dataset['facet'][vid][sid] and dataset['covarep'][vid][sid]:
            train_set_ids.append((vid, sid))

valid_set_ids = []
for vid in valid_ids:
    for sid in dataset['embeddings'][vid].keys():
        if dataset['embeddings'][vid][sid] and dataset['facet'][vid][sid] and dataset['covarep'][vid][sid]:
            valid_set_ids.append((vid, sid))

test_set_ids = []
for vid in test_ids:
    for sid in dataset['embeddings'][vid].keys():
        if dataset['embeddings'][vid][sid] and dataset['facet'][vid][sid] and dataset['covarep'][vid][sid]:
           test_set_ids.append((vid, sid))

# partition the training, valid and test set. all sequences will be padded/truncated to 15 steps
# data will have shape (dataset_size, max_len, feature_dim)
max_len = 15

train_set_audio = np.stack([pad(dataset['covarep'][vid][sid], max_len) for (vid, sid) in train_set_ids if dataset['covarep'][vid][sid]], axis=0)
valid_set_audio = np.stack([pad(dataset['covarep'][vid][sid], max_len) for (vid, sid) in valid_set_ids if dataset['covarep'][vid][sid]], axis=0)
test_set_audio = np.stack([pad(dataset['covarep'][vid][sid], max_len) for (vid, sid) in test_set_ids if dataset['covarep'][vid][sid]], axis=0)

train_set_visual = np.stack([pad(dataset['facet'][vid][sid], max_len) for (vid, sid) in train_set_ids], axis=0)
valid_set_visual = np.stack([pad(dataset['facet'][vid][sid], max_len) for (vid, sid) in valid_set_ids], axis=0)
test_set_visual = np.stack([pad(dataset['facet'][vid][sid], max_len) for (vid, sid) in test_set_ids], axis=0)

train_set_text = np.stack([pad(dataset['embeddings'][vid][sid], max_len) for (vid, sid) in train_set_ids], axis=0)
valid_set_text = np.stack([pad(dataset['embeddings'][vid][sid], max_len) for (vid, sid) in valid_set_ids], axis=0)
test_set_text = np.stack([pad(dataset['embeddings'][vid][sid], max_len) for (vid, sid) in test_set_ids], axis=0)

# sentiment scores
y_train = np.array([sentiments[vid][sid] for (vid, sid) in train_set_ids])
y_valid = np.array([sentiments[vid][sid] for (vid, sid) in valid_set_ids])
y_test = np.array([sentiments[vid][sid] for (vid, sid) in test_set_ids])

# normalize covarep and facet features, remove possible NaN values
visual_max = np.max(np.max(np.abs(train_set_visual), axis=0), axis=0)
visual_max[visual_max==0] = 1 # if the maximum is 0 we don't normalize this dimension
train_set_visual = train_set_visual / visual_max
valid_set_visual = valid_set_visual / visual_max
test_set_visual = test_set_visual / visual_max

train_set_visual[train_set_visual != train_set_visual] = 0
valid_set_visual[valid_set_visual != valid_set_visual] = 0
test_set_visual[test_set_visual != test_set_visual] = 0

audio_max = np.max(np.max(np.abs(train_set_audio), axis=0), axis=0)
train_set_audio = train_set_audio / audio_max
valid_set_audio = valid_set_audio / audio_max
test_set_audio = test_set_audio / audio_max

train_set_audio[train_set_audio != train_set_audio] = 0
valid_set_audio[valid_set_audio != valid_set_audio] = 0
test_set_audio[test_set_audio != test_set_audio] = 0

# early fusion: input level concatenation of features
x_train = np.concatenate((train_set_visual, train_set_audio, train_set_text), axis=2)
x_valid = np.concatenate((valid_set_visual, valid_set_audio, valid_set_text), axis=2)
x_test = np.concatenate((test_set_visual, test_set_audio, test_set_text), axis=2)

print("Data preprocessing finished! Begin compiling and training model.")

# Building FL fusion model
all_input = Input(shape=(maxlen,420), dtype='float32', name='input')
#FL_layer_0 = BatchNormalization(input_shape=(maxlen,420))(all_input)
FL_layer_1 = Dropout(0.2)(all_input)
FL_layer_2 = LSTM(128, return_sequences=False, trainable=True)(FL_layer_1)
FL_layer_3 = Dense(32, activation='relu', W_regularizer=l2(0.0), trainable=True)(FL_layer_2)
FL_layer_4 = Dense(32, activation='relu', W_regularizer=l2(0.0), trainable=True)(FL_layer_3)
FL_layer_5 = Dense(32, activation='relu', W_regularizer=l2(0.0), trainable=True)(FL_layer_4)
main_output = Dense(1, activation='tanh', W_regularizer=l2(0.01), name='main_output')(FL_layer_5) # valence regression
FL_model = Model(inputs=all_input, outputs=main_output)

# try using different optimizers and different optimizer configs
FL_model.compile(opt_func, loss_func, metrics=[pearson_cc,metr])	

print('Training...')
FL_model.fit(x_train,
          y_train,
          batch_size=batch_size,
          epochs=nb_epoch,
          validation_data=[x_valid, y_valid],
          callbacks=[early_stopping])

# Evaluation
print('\n\n\n\nEvaluating on train set...')
trn_score, trn_cc_emo, trn_mae_emo = FL_model.evaluate(x_train, y_train, batch_size=batch_size)
print('Valence Train cc:', trn_cc_emo)
print('Valence Train mae:', trn_mae_emo)
print('\nEvaluating on valisation set...')
val_score, val_cc_emo, val_mae_emo = FL_model.evaluate(x_valid, y_valid, batch_size=batch_size)
print('Valence Validation cc:', val_cc_emo)
print('Valence Validation mae:', val_mae_emo)
print('\nEvaluating on test set...')
tst_score, tst_cc_emo, tst_mae_emo = FL_model.evaluate(x_test, y_test, batch_size=batch_size)
print('Valence Test cc:', tst_cc_emo)
print('Valence Test mae:', tst_mae_emo)

# output predictions
np.set_printoptions(threshold=np.nan)
tst_pred_file = "prediction/pred_FL_uno.txt"
print('Printing predictions...')
tst_pred = FL_model.predict(x_test)
tst_df = pd.DataFrame(tst_pred)
tst_df.to_csv(tst_pred_file, index=False, header=False)

print('\nDone!')

# Flush outputs to log file
logger.flush()
logger.close()
