from lenet import LeNet
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras import backend as K 
import matplotlib.pyplot as plt 
import numpy as np

print("[INFO] accessing MNIST ")

((x_train, y_train), (x_test, y_test)) = mnist.load_data()
print(x_train.shape)

# check if channel first??
if K.image_data_format == "channels_first":
    # 6000 x 1 x 28 x 28
    x_train = x_train.reshape((x_train.shape[0],1,28,28))
    x_test = x_test.reshape((x_test.shape[0],1,28,28))
else:
    x_train = x_train.reshape((x_train.shape[0],28,28,1))
    x_test = x_test.reshape((x_test.shape[0],28,28,1))

# normalize to 0~1 
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0

# convert label to integer (one hot encoding)
le = LabelBinarizer()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)


print("[INFO] compiling model.....")
opt = SGD(lr=0.01)
model = LeNet.build(width=28,height=28,depth=1,classes=10)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])


# start training 
epoch = 5
bs = 128
print("[INFO] training network......")
H = model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=bs,epochs=epoch,verbose=1)


# evaluate the network 
print("[INFO] evaluating network....")
predictions = model.predict(x_test,batch_size=bs)
print(classification_report(y_test.argmax(axis=1),predictions.argmax(axis=1),target_names=[str(x) for x in le.classes_]))

# show Acc and Loss curve 
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,epoch),H.history["loss"],label='train_loss')
plt.plot(np.arange(0,epoch),H.history["val_loss"],label='val_loss')
plt.plot(np.arange(0,epoch),H.history["accuracy"],label='train_acc')
plt.plot(np.arange(0,epoch),H.history["val_accuracy"],label='val_acc')
plt.legend()
plt.show()

