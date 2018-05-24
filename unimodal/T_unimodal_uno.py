# Verbal unimodal singletask learning
# for ACL2018 Computational Modeling of Human Multimodal Language Workshop paper

from __future__ import print_function
import numpy as np
import pandas as pd
from collections import defaultdict
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Merge, Input
from keras.optimizers import RMSprop,Adamax
from keras.callbacks import EarlyStopping
from keras.regularizers import l1, l2
from keras import backend as K
from mmdata import MOSI

# turn off the warnings, be careful when use this
import warnings
warnings.filterwarnings("ignore")

# save outputs to a log file in case there is a broken pipe
import sys
idlestdout = sys.stdout
logger = open("prediction/output_T_unimodal_uno.txt", "w")
sys.stdout = logger

# custom evaluation metrics
def pearson_cc(y_true, y_pred):
    fsp = y_pred - K.mean(y_pred,axis=0)   
    fst = y_true - K.mean(y_true,axis=0) 
    devP = K.std(y_pred,axis=0)  
    devT = K.std(y_true,axis=0)

    return K.sum(K.mean(fsp*fst,axis=0)/(devP*devT))

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
embeddings = mosi.embeddings() # features
sentiments = mosi.sentiments() # Valence labels
train_ids = mosi.train()
valid_ids = mosi.valid()
test_ids = mosi.test()

# Some data preprocessing
x_train = []
y_train = []
x_valid = []
y_valid = []
x_test = []
y_test = []

print("Preparing train and test data...")
for vid, vdata in embeddings['embeddings'].items(): # note that even Dataset with one feature will require explicit indexing of features
    for sid, sdata in vdata.items():
        if sdata == []:
            continue
        example = []
        for i, time_step in enumerate(sdata):
            # data is truncated for 15 words
            if i == 15:
                break
            example.append(time_step[2]) # here first 2 dims (timestamps) will not be used

        for i in range(maxlen - len(sdata)):
            example.append(np.zeros(sdata[0][2].shape)) # padding each example to maxlen
        example = np.asarray(example)
        # Valence_label = 1 if sentiments[vid][sid] >= 0 else 0 # binarize the Valence labels
        Valence_label = sentiments[vid][sid] # Valence regression

        # train-validation-test partitions
        if vid in train_ids:
            x_train.append(example)
            y_train.append(Valence_label)
        elif vid in valid_ids:
            x_valid.append(example)
            y_valid.append(Valence_label)
        else:
            x_test.append(example)
            y_test.append(Valence_label)

# Prepare the final inputs as numpy arrays
x_train = np.asarray(x_train)
x_valid = np.asarray(x_valid)
x_test = np.asarray(x_test)
y_train = np.asarray(y_train)
y_valid = np.asarray(y_valid)
y_test = np.asarray(y_test)
print("Data preprocessing finished! Begin compiling and training model.")

#Building model
all_input = Input(shape=(maxlen, 300), dtype='float32', name='input')
h1 = LSTM(128, return_sequences=False, trainable=True)(all_input)
h2 = Dense(64, W_regularizer=l2(0.0), trainable=True)(h1)
main_output = Dense(1, activation='tanh', name='main_output')(h2) # valence regression
model = Model(inputs=all_input, outputs=main_output)

# try using different optimizers and different optimizer configs
model.compile(opt_func, loss_func, metrics=[pearson_cc,metr])

print('Training...')
model.fit(x_train,
          y_train,
          batch_size=batch_size,
          epochs=nb_epoch,
          validation_data=[x_valid, y_valid],
          callbacks=[early_stopping])

# Evaluation
print('\n\n\n\nEvaluating on train set...')
trn_score, trn_cc_emo, trn_mae_emo = model.evaluate(x_train, y_train, batch_size=batch_size)
print('Valence Train cc:', trn_cc_emo)
print('Valence Train mae:', trn_mae_emo)
print('\nEvaluating on valisation set...')
val_score, val_cc_emo, val_mae_emo = model.evaluate(x_valid, y_valid, batch_size=batch_size)
print('Valence Validation cc:', val_cc_emo)
print('Valence Validation mae:', val_mae_emo)
print('\nEvaluating on test set...')
tst_score, tst_cc_emo, tst_mae_emo = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Valence Test cc:', tst_cc_emo)
print('Valence Test mae:', tst_mae_emo)

# output predictions
np.set_printoptions(threshold=np.nan)
tst_pred_file = "prediction/pred_T_unimodal_uno.txt"
print('Printing predictions...')
tst_pred = model.predict(x_test)
tst_df = pd.DataFrame(tst_pred)
tst_df.to_csv(tst_pred_file, index=False, header=False)

print('\nDone!')

# Flush outputs to log file
logger.flush()
logger.close()
