# Large CNN model for the CIFAR-10 Dataset
import numpy
from keras.utils import np_utils
from keras import backend as K
from keras.datasets import cifar10
from keras.models import model_from_yaml

K.set_image_data_format('channels_first')

# fix random seed for reproducibility
seed = 3
numpy.random.seed(seed)

# load YAML and create model
yaml_file = open('models/model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
model = model_from_yaml(loaded_model_yaml)
# load weights into new model
model.load_weights("models/model.h5")

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-score[1]*100))