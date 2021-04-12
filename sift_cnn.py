import glob as gb
import shutil
import cv2
import glob
import numpy as np
import keras
import os

X_train = []
y_train = []
img_shape = (32, 32, 3)
trainpath = 'D:/datasets/crossvalidation/sift/geo/geo2/train/'
for folder in os.listdir(trainpath):
    folderclasspath = trainpath + folder
    for folderclass in os.listdir(folderclasspath):
        files = gb.glob(pathname=str(folderclasspath + '/' + folderclass + '/*'))
        for file in files:
            image = cv2.imread(file)

            X_train.append(list(image))

            y_train.append(int(folder))

X_test = []
y_test = []
img_shape = (32, 32, 3)
trainpath = 'D:/datasets/crossvalidation/sift/geo/geo2/test/'
for folder in os.listdir(trainpath):
    folderclasspath = trainpath + folder
    for folderclass in os.listdir(folderclasspath):
        files = gb.glob(pathname=str(folderclasspath + '/' + folderclass + '/*'))
        for file in files:
            image = cv2.imread(file)

            X_test.append(list(image))
            y_test.append(int(folder))


X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

KerasModel = keras.models.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(img_shape)),
    keras.layers.MaxPool2D(2, 2),

    # keras.layers.Dropout(rate=0.25) ,
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.BatchNormalization(),

    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),

    keras.layers.Dense(51, activation='softmax'),
])

KerasModel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print('Model Details are : ')
print(KerasModel.summary())

mcp1_save = keras.callbacks.ModelCheckpoint('mdl1_wts.hdf5', save_best_only=True, monitor='val_accuracy', mode='max')
mcp2_save = keras.callbacks.ModelCheckpoint('mdl2_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
early=keras.callbacks.EarlyStopping(monitor="val_accuracy",min_delta=0,patience=50,verbose=1,mode='max')
callbacklist=[mcp1_save,mcp2_save,early]
ThisModel = KerasModel.fit(X_train, y_train, epochs=150 , batch_size=1000,verbose=1, validation_data=(X_test, y_test),callbacks=callbacklist)


KerasModel.load_weights('mdl1_wts.hdf5')
y_resultat=KerasModel.predict(X_test)
trainpath = 'D:/datasets/crossvalidation/sift/geo/geo2/test/'
i=0
k=0
for folder in  os.listdir(trainpath ) :
    folderclasspath =  trainpath + folder
    for folderclass in os.listdir(folderclasspath):
        files = gb.glob(pathname= str( folderclasspath +'/'+ folderclass + '/*'))
        l=[]
        #for file in files:
            #l.append(np.argmax(y_resultat[i]))
            #i=i+1
        #if(max(l,key=l.count)==int(folder)):
            #k=k+1
        ki=[0]*51
        for file in files:
            ki=ki+y_resultat[i]
            i=i+1
        if(np.argmax(ki)==int(folder)):
            k=k+1
print(i,k)


KerasModel.evaluate(X_test, y_test)