"""
Train our RNN on extracted features or images.
"""
import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from data import DataSet
import time
import keras
import os.path
import argparse
#from tensorflow import keras
import keras.backend as K
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
#from tensorflow.keras.layers.recurrent import LSTM,GRU,SimpleRNN
from keras.layers import LSTM,GRU,SimpleRNN
import stadv
from stn import spatial_transformer_network as transformer
import scipy.misc
from keras.optimizers import Adam
from keras.losses import MSE,KLDivergence
#tf.config.run_functions_eagerly(True)
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_acc_metric = tf.keras.metrics.CategoricalAccuracy()
batch_size = 32
crop_size = 224
channels = 3
seq_len = 40
features = 2048
indicator = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
w_init = tf.random_normal_initializer()

rescale=[0.0,255.0]
mean=np.array([0.,0.,0.])
def lstm(inputs,input_shape,nb_classes):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model predomenently."""
        # Model.
        #model = Sequential()
        
        x = LSTM(2048, return_sequences=False,
                       input_shape=input_shape,
                       dropout=0.5)(inputs)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(nb_classes, activation='softmax')(x)
        return outputs



def train(path,data_type='features', seq_length=40, model='lstm', saved_model=None,
          class_limit=None, image_shape=None,
          load_to_memory=False, batch_size=32, nb_epoch=1):
    # Helper: Save the model.
    filepath=os.path.join('data', 'checkpoints', model + '-' + 'ori' + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5') 
    checkpointer = ModelCheckpoint(
        filepath=os.path.join('data', 'checkpoints', model + '-' + 'ori' + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5'),verbose=1,
        save_best_only=True)
    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join('data', 'logs', model))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=5)
    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('data', 'logs', model + '-' + 'trainingi2-' + \
        str(timestamp) + '.log'))

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(path,
            seq_length=seq_length,
            class_limit=class_limit
        )
    else:
        data = DataSet(path,
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        )

    # Get samples per epoch.
    # Multiply by:wq 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = (len(data.data) * 0.7) // batch_size
    
    if load_to_memory:
        # Get data.
        X, y = data.get_all_sequences_in_memory('train', data_type)
        print('X.shape',X.shape)
        X_test, y_test = data.get_all_sequences_in_memory('test', data_type)
        print('X_test shape is',X_test.shape)
    else:
        # Get generators.
        data_generator = data.frame_generator(batch_size, 'train', data_type)
        val_generator = data.frame_generator(batch_size, 'test', data_type)
      
    # Get the model.

    input_shape = (seq_length, 2048)
    inputs = keras.Input(shape=input_shape)
    outputs = lstm(inputs, input_shape,len(data.classes))
    model = keras.Model(inputs=inputs, outputs=outputs)
   
    optimizer = Adam(lr=1e-5, decay=1e-6)
    metrics = ['top_k_categorical_accuracy']
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                           metrics=metrics)
    #optimizer = keras.optimizers.SGD(learning_rate=1e-3)
    # Instantiate a loss function.
    #loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # Prepare the metrics.
    #train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    #val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    # Prepare the training dataset.
    batch_size = 32
    
    # Fit!
    if load_to_memory == True:
        model.fit(
            X,
            y,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger, checkpointer],
            epochs=nb_epoch)
    else:
        # Use fit generator.
        model.fit_generator(
            generator=data_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=nb_epoch,
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger, checkpointer],
            validation_data=val_generator,
            validation_steps=40,
            workers=2)

def main():
    """These are the main training settings. Set each before running
    this file."""
    parser = argparse.ArgumentParser(description='Use Adam optimizer to generate adversarial examples.')
    parser.add_argument('-i', '--input_dir', type=str, required=True,
                        help='Directory of dataset.')
    parser.set_defaults(use_crop=True)
    args = parser.parse_args()
    # Get the dataset.
    #print(args.input_dir)
    
    # model can be one of lstm, lrcn, mlp, conv_3d, c3d
    model = 'lstm'
    saved_model = None  # None or weights file
    class_limit = None  # int, can be 1-101 or None
    seq_length = 40
    load_to_memory = True  # pre-load the sequences into memory
    batch_size = 32
    nb_epoch = 500

    # Chose images or features and image shape based on network.
    if model in ['conv_3d', 'c3d', 'lrcn']:
        data_type = 'images'
        image_shape = (80, 80, 3)
    elif model in ['rnn','gru','lstm', 'mlp']:
        data_type = 'features'
        image_shape = None
    else:
        raise ValueError("Invalid model. See train.py for options.")

    train(args.input_dir,data_type, seq_length, model=model, saved_model=saved_model,
          class_limit=class_limit, image_shape=image_shape,
          load_to_memory=load_to_memory, batch_size=batch_size, nb_epoch=nb_epoch)

if __name__ == '__main__':
    main()
