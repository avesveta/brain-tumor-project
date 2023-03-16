# from ml_logic.data import save_nii_to_pkl
# from ml_logic.preprocessor import preprocess_split_model_1, try_path
# from ml_logic.model1 import glioma_label_unet_model, fit_complie_model
# from ml_logic.result import plot_loss, plot_confusion_matrix
# from ml_logic.parameters import cache_path

import os
import pandas as pd



channel = 't1'
#load nii file from cloud or local cach folder and save it as pkl in /raw_data


cwd = os.getcwd() #should be the /home/yaoyx001/code/avesveta/brain-tumor-project

pkl_file= f"Grade_ID_{channel}_nii.pkl"

pkl_path = os.path.join(cwd,'raw_data',pkl_file)

print('/home/yaoyx001/code/avesveta/brain-tumor-project/raw_data/Grade_ID_t1_nii.pkl')

df = pd.read_pickle('/home/yaoyx001/code/avesveta/brain-tumor-project/raw_data/Grade_ID_t1_nii.pkl')
