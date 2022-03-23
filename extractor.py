from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Input
import numpy as np
import keras
import PIL
from keras.layers import Dense, GlobalAveragePooling2D,Dropout
from keras.optimizers import SGD, Adam, RMSprop
def freeze_all_but_top(model):
    """Used to train just the top layers of the model."""
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in model.layers[:-2]:
        layer.trainable = False
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
class Customi3vModel(keras.Model):
    def train_step(self, data,step_size = 1/255,epsilon = 0.03,perturbed_steps = 10):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        #flows = tf.Variable(0.01*np.ones((x.shape[0],2, 224,224),dtype=np.float32))
        #modifier = tf.Variable(0.01*np.ones(x.shape,dtype=np.float32))
        flows = tf.random.uniform((x.shape[0],2, 224,224),-8/(seq_len*features),8/(seq_len*features))
        modifier = tf.random.uniform(x.shape,-8/(seq_len*features),8/(seq_len*features))
        with tf.GradientTape() as tape: 
            y_pred = self(x, training=True)  # Forward pass
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
def get_i3v_model(weights='imagenet'):
    # create the base pre-trained model
    base_model = InceptionV3(weights=weights, include_top=False)
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer
    predictions = Dense(101, activation='softmax')(x)
    # this is the model we will train
    #model = Model(inputs=base_model.input, outputs=predictions)
    return base_model.input,predictions
class Extractor():
    def __init__(self, weights='./inception-ori.017-1.68.hdf5'):
        """Either load pretrained from imagenet, or load our saved
        weights from our own training."""
        self.weights = weights  # so we can check elsewhere which model
        if weights is None:
            # Get model with pretrained weights.
            base_model = InceptionV3(
                weights='imagenet',
                include_top=True
            )
            # We'll extract features at the final pool layer.
            self.model = Model(
                inputs=base_model.input,
                outputs=base_model.get_layer('avg_pool').output
            )
        else:
            inputs,outputs = get_i3v_model()
            self.model = Customi3vModel(inputs, outputs)
            self.model = freeze_all_but_top(self.model)
            #model = load_model(ckpt_path+CKPT_PATHES[data_set_name+'_'+model_name])
            #model =get_model_i3(class_no)
            self.model.load_weights(weights)
            # Load the model first.
            #self.model = load_model(weights)
            # Then remove the top so we get features not predictions.
            # From: https://github.com/fchollet/keras/issues/2371
            self.model.layers.pop()
            self.model.layers.pop()  # two pops to get to pool layer
            self.model.outputs = [self.model.layers[-1].output]
            self.model.output_layers = [self.model.layers[-1]]
            self.model.layers[-1].outbound_nodes = []
    def extract(self, image_path):
        #try:
        img = image.load_img(image_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # Get the prediction.
        features = self.model.predict(x)
        if self.weights is None:
            # For imagenet/default network:
            features = features[0]
        else:
            # For loaded network:
            features = features[0]
        return features
      #except OSError:
        #return None
