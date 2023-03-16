import streamlit as st
import numpy as np
import nibabel as nib
import requests
#create a Streamlit app instance:
st.set_page_config(page_title='MRI Analyzer', page_icon=':brain:', layout='wide')
st.title('MRI Analyzer')

mri_file = st.file_uploader('Upload an MRI image (NIfTI format)', type=['nii'])

output = st.empty()

#Create a function that analyzes the uploaded MRI image using an API and displays the results:
def analyze_mri(mri_data):
    # Send the MRI data to the API
    api_url = 'https://myapi.com/analyze_mri'
    response = requests.post(api_url, data=mri_data)

    # Parse the API response and extract the analysis results
    result = response.json()
    tumor_risk = result['tumor_risk']

    # Display the analysis results
    output.text(f'The tumor is {tumor_risk} risk')

#Handle the user input
if mri_file is not None:
    # Load the MRI data from the uploaded file
    mri_data = nib.load(mri_file)

    # Analyze the MRI data and display the results
    #analyze_mri(mri_data)
    print('input is correct')
