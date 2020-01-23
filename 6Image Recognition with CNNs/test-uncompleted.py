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

#START CODING HERE----------------------------------------------------



#---------------------------------------------------------------------


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-score[1]*100))