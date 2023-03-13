1. Set env variable for GCP bucket in terminal
```
BUCKET_NAME=row_data_bucket
```

2. Upload directory (e.g. seg_UNET_baseline/model_glioma_seg_nii_UNET_baseline) to bucket
```
gsutil cp -r seg_UNET_baseline/model_glioma_seg_nii_UNET_baseline gs://$BUCKET_NAME
```
