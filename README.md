
#The code corresponds to an initial version and may differ in some aspects from the description in the paper. We will continue to update and refine the code in future releases.
# HSD-Sonar
## Environments
# Install all required packages
```bash
pip3 install -r requirements.txt
```

## Dataset
* FLSMDD  [https://drive.google.com/file/d/1zC5nObtLEOa1aqYj2Nz9ON9K6ML1WbPg/view](https://github.com/mvaldenegro/marine-debris-fls-datasets)
* SSSD [[Training]href="https://www.cis.jhu.edu/~kmei1/publics/shadow/datasets/aistd_train.zip [Testing]href="https://www.cis.jhu.edu/~kmei1/publics/shadow/datasets/istd_test.zip](https://link.springer.com/article/10.1007/s00371-025-03873-1#citeas)
* NNSSS  https://github.com/aburguera/NNSSS
* WSSSS https://github.com/YDY-andy/Sonar-dataset/tree/main
* AI4Shipwrecks https://umfieldrobotics.github.io/ai4shipwrecks/

## Training
1.You are expected to see the following file structres otherwise you need to manually rename those directory into the correct one.
``` 
DATA_FOLDER
├── train
│   ├── tr_1.png
│   ├── tr_2.png
│   └── ...
├── val
│   ├── val_1.png
│   ├── val_2.png
│   └── ...
└── test
    ├── ts_1.png
    ├── ts_2.png
    └── ...
```
2.Follow the instructions below to train our model. 
```bash
CUDA_VISIBLE_DEVICES={DEVICES} python3 main.py \
    --mode train \
    --model_type DDIM \
    --img_size {IMAGE_SIZE} \
    --num_img_channels {NUM_IMAGE_CHANNELS} \
    --dataset {DATASET_NAME} \
    --img_dir {DATA_FOLDER} \
    --train_batch_size 16 \
    --eval_batch_size 8 \
    --num_epochs 400
```
where:
- `DEVICES` is a comma-separated list of GPU device indices to use (e.g. `0,1,2,3`).
- `IMAGE_SIZE` and `NUM_IMAGE_CHANNELS` respectively specify the size of the images to train on (e.g. `256`) and the number of channels (`1` for greyscale, `3` for RGB).
- `model_type` specifies the type of diffusion model sampling algorithm to evaluate the model with, and can be `DDIM` or `DDPM`.
- `DATASET_NAME` is some name for your dataset (e.g. `breast_mri`).
- `DATA_FOLDER` is the path to your dataset directory, as outlined in the previous section.
- `--train_batch_size` and `--eval_batch_size` specify the batch sizes for training and evaluation, respectively. We use a train batch size of 16 for one 48 GB A6000 GPU for an image size of 256.
- `--num_epochs` specifies the number of epochs to train for (our default is 400).
## Testing
```bash
CUDA_VISIBLE_DEVICES={DEVICES} python3 main.py \
    --mode eval_many \
    --model_type DDIM \
    --img_size 256 \
    --num_img_channels {NUM_IMAGE_CHANNELS} \
    --dataset {DATASET_NAME} \
    --eval_batch_size 8 \
    --eval_sample_size 100
```


