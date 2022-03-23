"""
Train on images split into directories. This assumes we've split
our videos into frames and moved them to their respective folders.

Based on:
https://keras.io/preprocessing/image/
an
https://keras.io/applications/
"""
import os
import glob
import keras
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from data import DataSet
import PIL
import os.path
import argparse
from stn import spatial_transformer_network as transformer
parser = argparse.ArgumentParser(description='Use Adam optimizer to generate adversarial examples.')
parser.add_argument('-i', '--input_dir', type=str, required=True,
                        help='Directory of dataset.')
parser.set_defaults(use_crop=True)
args = parser.parse_args()
data = DataSet(args.input_dir)

# Helper: Save the model.
checkpointer = ModelCheckpoint(
    filepath=os.path.join('data', 'checkpoints', 'inception-pgd.{epoch:03d}-{val_loss:.2f}.hdf5'),
    verbose=1,
    save_best_only=False)

# Helper: Stop when we stop learning.
early_stopper = EarlyStopping(patience=10)

# Helper: TensorBoard
tensorboard = TensorBoard(log_dir=os.path.join('data', 'logs'))

def get_generators(data_file):
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        horizontal_flip=True,
        rotation_range=10.,
        width_shift_range=0.2,
        height_shift_range=0.2)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_file,'data', 'train','train'),
        target_size=(299, 299),
        batch_size=32,
        classes=data.classes,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        os.path.join(data_file,'data', 'test','test'),
        target_size=(299, 299),
        batch_size=32,
        classes=data.classes,
        class_mode='categorical')

    return train_generator, validation_generator

def get_model(weights='imagenet'):
    # create the base pre-trained model
    base_model = InceptionV3(weights=weights, include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer
    predictions = Dense(len(data.classes), activation='softmax')(x)

    # this is the model we will train
    #model = Model(inputs=base_model.input, outputs=predictions)
    return base_model.input,predictions

def freeze_all_but_top(model):
    """Used to train just the top layers of the model."""
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in model.layers[:-2]:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def freeze_all_but_mid_and_top(model):
    """After we fine-tune the dense layers, train deeper."""
    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 172 layers and unfreeze the rest:
    for layer in model.layers[:172]:
        layer.trainable = False
    for layer in model.layers[172:]:
        layer.trainable = True
    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(
        optimizer=SGD(lr=0.0001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy'])
    return model
def train_model(model, nb_epoch, generators, callbacks=[]):
    train_generator, validation_generator = generators
    model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        validation_data=validation_generator,
        validation_steps=10,
        epochs=nb_epoch,
        callbacks=callbacks)
    return model

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
                y_pred = self(x_adv, training=False) 
                loss = self.compiled_loss(y, self(x_adv), regularization_losses=self.losses)
            grad = tape.gradient(loss,x_adv)
            x_adv = x_adv + step_size* tf.sign(grad)    
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
def main(weights_file,data_file):
    #model = get_model()
    new_file = os.path.join(data_file,'data', 'train')
    filelist=os.listdir(new_file)
    for files in filelist:
        imgs_ = glob.glob(new_file + "/" + files+"/*.jpg")
        for img in imgs_:
            try:
                img = PIL.Image.open(img)
            except PIL.UnidentifiedImageError:
                print(img)
                os.remove(img)
    inputs,outputs = get_model()
    model = CustomModel(inputs, outputs)
    generators = get_generators(data_file)
    if weights_file is None:
        print("Loading network from ImageNet weights.")
        # Get and train the top layers.
        model = freeze_all_but_top(model)
        try:
            model = train_model(model, 10, generators)
        except:
            pass
    else:
        print("Loading saved model: %s." % weights_file)
        model.load_weights(weights_file)

    # Get and train the mid layers.
    model = freeze_all_but_mid_and_top(model)
    model = train_model(model, 1000, generators,
                        [checkpointer, early_stopper, tensorboard])

if __name__ == '__main__':
    weights_file = None
    #parser = argparse.ArgumentParser(description='Use Adam optimizer to generate adversarial examples.')
    #parser.add_argument('-i', '--input_dir', type=str, required=True,
     #                   help='Directory of dataset.')
    #parser.set_defaults(use_crop=True)
    #args = parser.parse_args()
    main(weights_file,args.input_dir)
