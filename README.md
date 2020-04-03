# iMet-Collection-2019-FGVC-6_54th
**The solution to iMet-Collection-2019 FGVC 6**<br><br>
The detailed description is here:<br>
https://www.kaggle.com/c/imet-2019-fgvc6<br>

## detail
* model: SEResNext50,SEResNext101 
* batchsize:64
* input_size:320
* optimizer:Adam
* epochs:20
* lr:epoch<18 0.0001 20<0.00005
* Final model:5fold+SEResNext50x0.2+SEResNext50x0.8 
* Private Leader board 0.631

## How to run
1.config.py <br>
* fold=0,1,2,3,4
* model_name=resnet50,resnet101
### train
* python imet_train.py
### predict
* python predict.py

## Requirements
* numpy==1.18.1
* opencv==4.1.2.30
* Pillow==6.2.2
* scikit-image==0.16.2
* scikit-learn==0.20.1
* torch==1.1.0
* torchvision==0.3.0
