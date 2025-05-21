# One-UViCS README
[Uploading 整体架构图new.png…]
```
conda create --name memsam python=3.10
conda activate memsam
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118!()

pip install requirements.txt
```

## Usage
### prepare dataset
First, download the dataset from data service:

- Private Breast Data

- [Abnormcardiacechovideos](https://www.kaggle.com/datasets/xiaoweixumedicalai/abnormcardiacechovideos)

- [EchoNet-Dynamic](https://echonet.github.io/dynamic/index.html)

Then process the dataset according to `utils/preprocess_****.py` 

pretrain checkpoint download

[ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

### train and test
Use `train_video.py` and `test_video.py` to train and test separately.

### evaluation

evaluation.py, The default 1351 line evaw_slice function can obtain relevant performance. Annotated evad_slice2 yields ROC curve and confusion matrix.

## Acknowledgement
The work is based on MemSAM. Thanks for the open source contributions to these efforts!

