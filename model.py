import os
import cv2
import numpy as np

labels = ['angry', 'disguist', 'happy', 'sad', 'surprise']
img_size = 224

def get_training_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return data


train = get_training_data('train_crop')
data = np.array(train, dtype=object)  # Convert the list of lists to a NumPy array



#train = get_training_data('/content/drive/MyDrive/train_crop'
x=[]
y=[]
for i,j in train:
  x.append(i)
  y.append(j)

import numpy as np
x=np.array(x)
y=np.array(y)

#normalize the data
x=x/255.0

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=True,test_size=0.2)

'''
It appears that you want to convert the labels in y_train from a shape of (None, 1) to (None, 5) for a multi-class classification task with 5
classes. To achieve this, you can use the to_categorical function from TensorFlow/Keras. However, there is a prerequisite: your original
labels in y_train need to be in integer format and not one-hot encoded.

Here's how you can convert y_train to (None, 5):
'''
#print(y_train)
from tensorflow.keras.utils import to_categorical
# Assuming y_train contains integer labels (0 to 4 for 5 classes)
y_train= to_categorical(y_train, num_classes=5)

from tensorflow.keras.utils import to_categorical
# Assuming y_train contains integer labels (0 to 4 for 5 classes)
y_test= to_categorical(y_test, num_classes=5)

# Working with pre trained model
#MobileNet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import utils
import os
from keras.layers import Flatten, Dense
from keras.models import Model
from tensorflow.keras.utils import img_to_array,load_img
from keras.preprocessing.image import  ImageDataGenerator
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.losses import categorical_crossentropy
base_model = MobileNet( input_shape=(224,224,3), include_top= False )

for layer in base_model.layers:
  layer.trainable = False


x = Flatten()(base_model.output)


x = Dense(units=5 , activation='softmax' )(x) #change if needed 6 classes are present

# creating our model.
model = Model(base_model.input, x)
#model = Model(inputs=base_model.input, outputs=output_layer)


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']  )

x_tr=x_train.reshape(598,224,224,3)
x_tt=x_test.reshape(150,224,224,3)

#MobileNet - without random.shuffle
mm= model.fit(x_tr, y_train, epochs=20, validation_data=(x_tt, y_test))

model.save('emotion_model_mn.h5')

# Evaluate the model on your test data.
test_loss, test_accuracy = model.evaluate(x_tt, y_test)
print(f"Test accuracy: {test_accuracy*100:.2f}%")