# Tensor Fusion Network multimodal multitask learning (valence+intensity)
# for ACL2018 Computational Modeling of Human Multimodal Language Workshop paper

from __future__ import print_function
import numpy as np
import pandas as pd
from collections import defaultdict
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Merge, Input, BatchNormalization, Flatten, Reshape, merge, concatenate
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
if len(sys.argv) == 2:
    output_dir = sys.argv[1]
else:
    raise NameError('Please provide an output directory, e.g.\n'
        '/ACL2018/prediction')

idlestdout = sys.stdout
logger = open(output_dir + "/output_TFN_int.txt", "w")
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
weight_main = 1.0 # weight for multitask learning
# for Valence intensity classification
loss_func_aux = 'categorical_crossentropy' # loss function
metr_aux = 'accuracy' # evaluation metric
weight_aux = 0.5 # weight for multitask learning

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

x_A_train = np.stack([pad(dataset['covarep'][vid][sid], max_len) for (vid, sid) in train_set_ids if dataset['covarep'][vid][sid]], axis=0)
x_A_valid = np.stack([pad(dataset['covarep'][vid][sid], max_len) for (vid, sid) in valid_set_ids if dataset['covarep'][vid][sid]], axis=0)
x_A_test = np.stack([pad(dataset['covarep'][vid][sid], max_len) for (vid, sid) in test_set_ids if dataset['covarep'][vid][sid]], axis=0)

x_V_train = np.stack([pad(dataset['facet'][vid][sid], max_len) for (vid, sid) in train_set_ids], axis=0)
x_V_valid = np.stack([pad(dataset['facet'][vid][sid], max_len) for (vid, sid) in valid_set_ids], axis=0)
x_V_test = np.stack([pad(dataset['facet'][vid][sid], max_len) for (vid, sid) in test_set_ids], axis=0)

x_T_train = np.stack([pad(dataset['embeddings'][vid][sid], max_len) for (vid, sid) in train_set_ids], axis=0)
x_T_valid = np.stack([pad(dataset['embeddings'][vid][sid], max_len) for (vid, sid) in valid_set_ids], axis=0)
x_T_test = np.stack([pad(dataset['embeddings'][vid][sid], max_len) for (vid, sid) in test_set_ids], axis=0)

# sentiment scores
y_train = np.array([sentiments[vid][sid] for (vid, sid) in train_set_ids])
y_valid = np.array([sentiments[vid][sid] for (vid, sid) in valid_set_ids])
y_test = np.array([sentiments[vid][sid] for (vid, sid) in test_set_ids])

# intensity classes
z_train = []
for (vid, sid) in train_set_ids:
    if abs(sentiments[vid][sid]) >= 2.5:
        Intensity_label = [0,0,0,1] # strong
    elif abs(sentiments[vid][sid]) >= 1.5:
        Intensity_label = [0,0,1,0] # medium
    elif abs(sentiments[vid][sid]) >= 0.5:
        Intensity_label = [0,1,0,0] # weak
    else:
        Intensity_label = [1,0,0,0] # neutral
    z_train.append(Intensity_label)

z_valid = []
for (vid, sid) in valid_set_ids:
    if abs(sentiments[vid][sid]) >= 2.5:
        Intensity_label = [0,0,0,1] # strong
    elif abs(sentiments[vid][sid]) >= 1.5:
        Intensity_label = [0,0,1,0] # medium
    elif abs(sentiments[vid][sid]) >= 0.5:
        Intensity_label = [0,1,0,0] # weak
    else:
        Intensity_label = [1,0,0,0] # neutral
    z_valid.append(Intensity_label)

z_test = []
for (vid, sid) in test_set_ids:
    if abs(sentiments[vid][sid]) >= 2.5:
        Intensity_label = [0,0,0,1] # strong
    elif abs(sentiments[vid][sid]) >= 1.5:
        Intensity_label = [0,0,1,0] # medium
    elif abs(sentiments[vid][sid]) >= 0.5:
        Intensity_label = [0,1,0,0] # weak
    else:
        Intensity_label = [1,0,0,0] # neutral
    z_test.append(Intensity_label)

z_train = np.asarray(z_train)
z_valid = np.asarray(z_valid)
z_test = np.asarray(z_test)

# normalize covarep and facet features, remove possible NaN values
visual_max = np.max(np.max(np.abs(x_V_train), axis=0), axis=0)
visual_max[visual_max==0] = 1 # if the maximum is 0 we don't normalize this dimension
x_V_train = x_V_train / visual_max
x_V_valid = x_V_valid / visual_max
x_V_test = x_V_test / visual_max

x_V_train[x_V_train != x_V_train] = 0
x_V_valid[x_V_valid != x_V_valid] = 0
x_V_test[x_V_test != x_V_test] = 0

audio_max = np.max(np.max(np.abs(x_A_train), axis=0), axis=0)
x_A_train = x_A_train / audio_max
x_A_valid = x_A_valid / audio_max
x_A_test = x_A_test / audio_max

x_A_train[x_A_train != x_A_train] = 0
x_A_valid[x_A_valid != x_A_valid] = 0
x_A_test[x_A_test != x_A_test] = 0

print("Data preprocessing finished! Begin compiling and training model.")

# Building model
# Vocal
covarep_layer_0 = Input(shape=(maxlen,74), dtype='float32', name = 'covarep_layer_0')
#covarep_layer_1 = BatchNormalization(input_shape=(maxlen,74))(covarep_layer_0)
covarep_layer_2 = Dropout(0.2)(covarep_layer_0)
covarep_layer_3 = Dense(32, activation='relu', W_regularizer=l2(0.0), trainable=True)(covarep_layer_2)
covarep_layer_4 = Dense(32, activation='relu', W_regularizer=l2(0.0), trainable=True)(covarep_layer_3)
covarep_layer_5 = Dense(32, activation='relu', W_regularizer=l2(0.0), trainable=True)(covarep_layer_4)
covarep_layer_6 = Reshape((15, 32))(covarep_layer_5)

# Visual
facet_layer_0 = Input(shape=(maxlen,46), dtype='float32', name = 'facet_layer_0')
#facet_layer_1 = BatchNormalization(input_shape=(maxlen,46))(facet_layer_0)
facet_layer_2 = Dropout(0.2)(facet_layer_0)
facet_layer_3 = Dense(32, activation='relu', W_regularizer=l2(0.0), trainable=True)(facet_layer_2)
facet_layer_4 = Dense(32, activation='relu', W_regularizer=l2(0.0), trainable=True)(facet_layer_3)
facet_layer_5 = Dense(32, activation='relu', W_regularizer=l2(0.0), trainable=True)(facet_layer_4)
facet_layer_6 = Reshape((15, 32))(facet_layer_5)

# Verbal
text_layer_0 = Input(shape=(maxlen, 300), dtype='float32', name='text_layer_0')
#text_layer_1 = BatchNormalization(input_shape=(maxlen,300))(text_layer_0)
text_layer_2 = LSTM(128, return_sequences=True, trainable=True)(text_layer_0)
text_layer_3 = Dense(64, activation='relu', W_regularizer=l2(0.0), trainable=True)(text_layer_2)
text_layer_4 = Reshape((1, 15 * 64))(text_layer_3)

# Modality fusion - TFN
dot_layer1 = merge([covarep_layer_6, facet_layer_6], mode='dot', dot_axes=1, name='dot_layer1')
dot_layer1_reshape = Reshape((1, 32 * 32), name='dot_layer1_reshape')(dot_layer1)
dot_layer2 = merge([dot_layer1_reshape, text_layer_4], mode='dot', dot_axes=1, name='dot_layer2')
TFN_layer_0 = Reshape((15, 32 * 32 * 64), name='TFN_layer_0')(dot_layer2)
TFN_layer_1 = Dropout(0.2)(TFN_layer_0)
TFN_layer_2 = LSTM(128, return_sequences=False, trainable=True)(TFN_layer_1)
TFN_layer_3 = Dense(32, activation='relu', W_regularizer=l2(0.01))(TFN_layer_2)
TFN_layer_4 = Dense(32, activation='relu', W_regularizer=l2(0.01))(TFN_layer_3)
TFN_layer_5 = Dense(32, activation='relu', W_regularizer=l2(0.01))(TFN_layer_4)
#TFN_layer_5 = Flatten()
main_output = Dense(1, activation='tanh', W_regularizer=l2(0.01), name='main_output')(TFN_layer_5) # valence regression
auxiliary_output = Dense(4, activation='softmax', name='aux_output')(TFN_layer_5) # Intensity classification
TFN_model = Model(inputs=[covarep_layer_0, facet_layer_0, text_layer_0], outputs=[main_output, auxiliary_output])

# try using different optimizers and different optimizer configs
TFN_model.compile(optimizer=opt_func,
              loss={'main_output': loss_func, 'aux_output': loss_func_aux},
              loss_weights={'main_output': weight_main, 'aux_output': weight_aux},
              metrics={'main_output': [pearson_cc,metr], 'aux_output': metr_aux})	

print('Training...')
TFN_model.fit([x_A_train, x_V_train, x_T_train],
          {'main_output': y_train, 'aux_output': z_train},
          batch_size=batch_size,
          epochs=nb_epoch,
          validation_data=[[x_A_valid, x_V_valid, x_T_valid], {'main_output': y_valid, 'aux_output': z_valid}],
          callbacks=[early_stopping])

# Evaluation
print('\n\n\n\nEvaluating on train set...')
trn_score, trn_score_emo, trn_score_v, trn_cc_emo, trn_mae_emo, trn_mae_v = TFN_model.evaluate([x_A_train, x_V_train, x_T_train], {'main_output': y_train, 'aux_output': z_train}, batch_size=batch_size)
print('Valence Train cc:', trn_cc_emo)
print('Valence Train mae:', trn_mae_emo)
print('\nEvaluating on valisation set...')
val_score, val_score_emo, val_score_v, val_cc_emo, val_mae_emo, val_mae_v = TFN_model.evaluate([x_A_valid, x_V_valid, x_T_valid], {'main_output': y_valid, 'aux_output': z_valid}, batch_size=batch_size)
print('Valence Validation cc:', val_cc_emo)
print('Valence Validation mae:', val_mae_emo)
print('\nEvaluating on test set...')
tst_score, tst_score_emo, tst_score_v, tst_cc_emo, tst_mae_emo, tst_mae_v = TFN_model.evaluate([x_A_test, x_V_test, x_T_test], {'main_output': y_test, 'aux_output': z_test}, batch_size=batch_size)
print('Valence Test cc:', tst_cc_emo)
print('Valence Test mae:', tst_mae_emo)

# output predictions
np.set_printoptions(threshold=np.nan)
tst_pred_file = output_dir + "/pred_TFN_int.txt"
print('Printing predictions...')
tst_pred = TFN_model.predict([x_A_test, x_V_test, x_T_test])
tst_df = pd.DataFrame(tst_pred[0])
tst_df.to_csv(tst_pred_file, index=False, header=False)

print('\nDone!')

# Flush outputs to log file
logger.flush()
logger.close()
