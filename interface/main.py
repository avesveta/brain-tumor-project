from ml_logic.data import save_nii_to_pkl
from ml_logic.preprocessor import preprocess_split_model_1
from ml_logic.model1 import glioma_label_unet_model, fit_complie_model
from ml_logic.result import plot_loss, plot_confusion_matrix
from ml_logic.parameters import cache_path
import os


channel = 't1'
#load nii file from cloud or local cach folder and save it as pkl in /raw_data


cwd = os.getcwd() #should be the /home/yaoyx001/code/avesveta/brain-tumor-project
pkl_file= f"Grade_ID_{channel}_nii.pkl"
pkl_path = os.path.join(cwd,'raw_data',pkl_file)

print('loading nii from cloud as save it in raw_data folder')
if not os.path.isfile(pkl_path):
    save_nii_to_pkl(cache_path,pkl_path,channel)
    print(f'{pkl_file} is saved into raw_data')
else:
    print('pkl is already in raw_data')


#preprocess the data and split it in Train Test
print(f'preprocess the {pkl_file} and split in Train-Test') # TODO
X_train, X_test, y_train, y_test=preprocess_split_model_1(pkl_path,channel)
print('Train-Test split is done')

#3D Unet model
print('making a model')
model = glioma_label_unet_model()
model.summary()

#train the model
print('fiting the model')
history = fit_complie_model(model,X_train,y_train)

#save the model in current folder

model.save(f'saved_models/model_glioma_{channel}_nii_3dUnet')
print(f'model saved in saved_models/model_glioma_{channel}_nii_3dUnet')
#show the result
plot_loss(history)

#show the confusion matrix
plot_confusion_matrix(model,X_test,y_test)
