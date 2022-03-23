'''
Loads pretrained model of I3d Inception architecture for the paper: 'https://arxiv.org/abs/1705.07750'
Evaluates a RGB and Flow sample similar to the paper's github repo: 'https://github.com/deepmind/kinetics-i3d'
'''

import numpy as np
import argparse
import keras
import os.path
import tensorflow as tf
from keras.layers import Dropout,Dense
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from data import DataSet
from i3d_inception import Inception_Inflated3d
import stadv
import time
from stn import spatial_transformer_network as transformer
import scipy.misc
from keras.optimizers import Adam
from keras.losses import MSE,KLDivergence
NUM_FRAMES = 40
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
NUM_RGB_CHANNELS = 3
NUM_FLOW_CHANNELS = 2
batch_size = 32
crop_size = 224
channels = 3
seq_len = 40
features = 2048
NUM_CLASSES = 400

rescale=[0.0,255.0]
mean=np.array([0.,0.,0.])
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_acc_metric = tf.keras.metrics.CategoricalAccuracy()
SAMPLE_DATA_PATH = {
    'rgb' : 'data/v_CricketShot_g04_c01_rgb.npy',
    'flow' : 'data/v_CricketShot_g04_c01_flow.npy'
}

LABEL_MAP_PATH = 'data/label_map.txt'
def I3D(inputs):
    rgb_model = Inception_Inflated3d(
                include_top=True,
                weights='rgb_kinetics_only',
                input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
                classes=NUM_CLASSES)
    x = rgb_model(inputs)
    x = Dropout(0.8)(x)
    predictions = Dense(101, activation='softmax')(x)
    return predictions

    
def train(path,data_type='features', seq_length=40, model='lstm', saved_model=None,
          class_limit=None, image_shape=None,
          load_to_memory=False, batch_size=32, nb_epoch=1):
    # load the kinetics classes
    filepath=os.path.join('data', 'checkpoints', model + '-' + 'pgd' + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5') 
    checkpointer = ModelCheckpoint(
        filepath=os.path.join('data', 'checkpoints', model + '-' + 'pgd' + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5'),verbose=1,
        save_best_only=False)
    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join('data', 'logs', model))
    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=10)
    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('data', 'logs', model + '-' + 'trainingi2-' + \
        str(timestamp) + '.log'))

    # Get the data and process it.
    
    data = DataSet(path,
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        )
    steps_per_epoch = (len(data.data) * 0.7) // batch_size
    data_generator = data.frame_generator(path,batch_size, 'train', data_type)
    val_generator = data.frame_generator(path,batch_size, 'test', data_type)
    #kinetics_classes = [x.strip() for x in open(LABEL_MAP_PATH, 'r')]
    class CustomModel(keras.Model):
      def train_step(self, data,step_size = 1/255,epsilon = 0.03,perturbed_steps = 10):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        
        #flows = tf.Variable(0.01*np.ones((x.shape[0],2, 224,224),dtype=np.float32))
        #modifier = tf.Variable(0.01*np.ones(x.shape,dtype=np.float32))
        flows = tf.random.uniform((x.shape[0],2, 224,224),-8/(seq_len*features),8/(seq_len*features))
        modifier = tf.random.uniform(x.shape,-8/(seq_len*features),8/(seq_len*features))
        rescale=[0.0,255.0]
        mean=np.array([0.,0.,0.])
  
        
        noise = tf.random.uniform(x.shape,-8/(seq_len*features),8/(seq_len*features))
        x_adv = x + 0.001 * noise
        #x_adv = tf.minimum(tf.maximum(stadv.layers.flow_st((x+modifier)*255.0, flows, 'NHWC'), -mean+rescale[0]), -mean+rescale[1])/255.0
        for _ in range(perturbed_steps):
            with tf.GradientTape(persistent=True) as tape:
                x_adv = tf.minimum(tf.maximum(stadv.layers.flow_st((x_adv+modifier)*255.0, flows, 'NHWC'), -mean+rescale[0]), -mean+rescale[1])/255.0
                #x_adv = tf.minimum(tf.maximum(modifier+perturbed_images*255.0, -spec.mean+spec.rescale[0]), -spec.mean+spec.rescale[1])/255.0
                y_pred = self(x_adv, training=False) 
                loss = self.compiled_loss(y, self(x_adv), regularization_losses=self.losses)
            grad = tape.gradient(loss,x_adv)
            x_adv = x_adv + step_size* tf.sign(grad)
            #flows = flows + step_size* tf.sign(grad)
            #modifier = modifier + step_size* tf.sign(grad)
            #self.optimizer.minimize(loss,[modifier,flows])
        with tf.GradientTape() as tape: 
            y_pred = self(x_adv, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss_1 = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Compute gradients
        trainable_vars = self.trainable_variables
        
        loss = loss_1 
       
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        
        return {m.name: m.result() for m in self.metrics} 
    batch_size = 32
    input_shape = (NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS)
    inputs = keras.Input(shape=input_shape)
    outputs = I3D(inputs)
    #model = keras.Model(inputs=inputs, outputs=outputs)
    model = CustomModel(inputs, outputs)
    optimizer = Adam(lr=1e-5, decay=1e-6)
    metrics = ['top_k_categorical_accuracy']
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                           metrics=metrics)
    # produce final model logits
    #model.load_weights('../UCF101/checkpoints/I3Dadv/i3d-ori.021-0.818.hdf5')
    #model.load_weights('data/checkpoints/i3d-adv-pgd.007-2.490.hdf5')
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
    
   
    model = 'i3d'
    saved_model = None  # None or weights file
    class_limit = None  # int, can be 1-101 or None
    seq_length = 40
    load_to_memory = False  # pre-load the sequences into memory
    batch_size = 4
    nb_epoch = 500

    # Chose images or features and image shape based on network.
    
    data_type = 'images'

    train(args.input_dir,data_type, seq_length, model=model, saved_model=saved_model,
          class_limit=class_limit, image_shape=image_shape,
          load_to_memory=load_to_memory, batch_size=batch_size, nb_epoch=nb_epoch)

if __name__ == '__main__':
    main()


