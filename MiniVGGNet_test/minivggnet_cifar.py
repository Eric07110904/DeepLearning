import matplotlib.pyplot as plt 
import numpy as np 
import argparse
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from minivggnet import MiniVGGNet 
from tensorflow.keras.optimizers import SGD 
from tensorflow.keras.datasets import cifar10

# ap = argparse.ArgumentParser()
# ap.add_argument("-o","--output",required=True,help="path to the output loss/acc plot")
# args = vars(ap.parse_args())


# load cifar-10 datasets
print("[INFO] loading CIFAR-10 data...")
((x_train,y_train),(x_test,y_test)) = cifar10.load_data()

print(y_train[:5,0])

# normalize image 
x_train = x_train.astype("float32")/255.0
x_test = x_test.astype("float32")/255.0 

# label encoder (one hot )
le = LabelBinarizer()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)
labelNames = ["airplane","automobile", "bird", "cat", "deer","dog", "frog", "horse", "ship", "truck"]

print("[INFO] compiling model...")
opt = SGD(learning_rate=0.01,decay=0.01/40,momentum=0.9,nesterov=True)
model = MiniVGGNet.build(width=32,height=32,depth=3,classes=10)
model.compile(optimizer=opt,loss="categorical_crossentropy",metrics=["accuracy"])

# start training the net
print("[INFO] training network...")
H = model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=64,epochs=40,verbose=1)
