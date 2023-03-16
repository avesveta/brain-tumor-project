from google.cloud import storage
import os
import nibabel as nib
import pandas as pd
import numpy as np


def load_nii_from_gcp(filename:str,cache_folder_path):
    r"""Load file given filename, from GCP or direktly from cache_folder

    Parameters
    ----------
    filename : str or os.PathLike
       specification of file to load
    \*\*kwargs : keyword arguments
        Keyword arguments to format-specific load

    Returns
    -------
    img : ``SpatialImage``
       Image of guessed type
    """


    #make the connection to GCP
    client = storage.Client()
    #get the Bucket Name from .env
    bucket_name = os.getenv('BUCKET_NAME')
    #set bucket
    bucket = client.get_bucket(bucket_name)
    #give the blob_name that you want to access
    blob_name= filename
    #get the blob(file)
    blob = bucket.blob(blob_name)

    cache_file_path = os.path.join(cache_folder_path, blob_name)
    #save the file in cache_folder
    if not os.path.isfile(cache_file_path):
        blob.download_to_filename(cache_file_path)

    img = nib.load(cache_file_path)
    return img

def save_nii_to_pkl(cache_folder_path,pkl_path,channel):
    # Path
    path = cache_folder_path

    #upload channel
    channel # t1,t2,flair,t1ce,seg
    #
    df = pd.read_csv('raw_data/name_mapping_2020.csv')
    df = df.drop(columns=['BraTS_2017_subject_ID', 'BraTS_2018_subject_ID',
                            'BraTS_2019_subject_ID', 'TCGA_TCIA_subject_ID'])
    df['Grade'].value_counts()
    df[f'{channel}_nii']= df['BraTS_2020_subject_ID'].apply(lambda x: np.array(load_nii_from_gcp(x+f'_{channel}.nii',path).dataobj))
    df.to_pickle(pkl_path)
