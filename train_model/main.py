from ml_logic.data import save_nii_to_pkl
from ml_logic.preprocessor import preprocess_split_model_1
from ml_logic.model1 import glioma_label_unet_model
from ml_logic.result import plot_loss, plot_confusion_matrix
from ml_logic.parameters import cache_path
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import os
import numpy as np
import tensorflow as tf

channel = 't1'
#load nii file from cloud or local cach folder and save it as pkl in /raw_data


cwd = os.getcwd() #should be the /home/yaoyx001/code/avesveta/brain-tumor-project
pkl_file= f"Grade_ID_{channel}_nii.pkl"
pkl_path = os.path.join(cwd,'raw_data',pkl_file)
model_files = f'model_glioma_{channel}_nii_3dUnet.h5'
model_path = os.path.join(cwd,'saved_models',model_files)

print('loading nii from cloud as save it in raw_data folder')
if not os.path.isfile(pkl_path):
    save_nii_to_pkl(cache_path,pkl_path,channel)
    print(f'{pkl_file} is saved into raw_data')
else:
    print('pkl is already in raw_data')


#preprocess the data and split it in Train Test
print(f'preprocess the {pkl_file} and split in Train-Test')
X_train, X_test, y_train, y_test=preprocess_split_model_1(pkl_path,channel)

print('Train-Test split is done')
print('shape of X_train:')
print(X_train.shape)
print('sample in y_train:')
print(np.unique(y_train,return_counts=True))

#check if the saved_models exits, if not, make it
if not os.path.isfile(os.path.join(cwd,'saved_models')):
        os.mkdir(os.path.join(cwd,'saved_models'))
        print('saved_models is created')
#3D Unet model
if not os.path.exists(model_path):#check if the model.h5 file exits, if not, train it
    print('making a model')
    model = glioma_label_unet_model()
    model.summary()
    #add earlystop
    es = EarlyStopping(patience=5, restore_best_weights = True
                       ,monitor='loss', min_delta=0.01)

    #set optimizer with learning rate
    optim=Adam(learning_rate= 0.001)
    #complie the model
    model.compile(loss = 'binary_crossentropy',
                  optimizer = optim,
                  metrics=['accuracy'])
    #train the model
    print('fiting the model')
    history = model.fit(X_train, y_train,
                        epochs = 30,
                        batch_size = 4,
                        callbacks = [es],
                        validation_data=(X_test, y_test),
                        shuffle =True,
                        verbose = 1)

    #save the model in current folder
    model.save(model_path)
    print(f'model saved in saved_models/model_glioma_{channel}_nii_3dUnet.h5')
    #show the result
    plot_loss(history)
else:
    model = tf.keras.models.load_model(model_path)

#show the confusion matrix
plot_confusion_matrix(model,X_test,y_test)
