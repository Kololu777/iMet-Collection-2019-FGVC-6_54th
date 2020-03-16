# iMet-Collection-2019-FGVC-6_54th
**The solution t0 iMet-Collection-2019 FGVC 6**<br><br>
The detailed description is here:<br>
https://www.kaggle.com/c/imet-2019-fgvc6<br>

model: SEResNext50,SEResNext101 <br>
batchsize:64<br>
input_size:320<br>
optimizer:Adam<br>
epochs:20<br>
lr:epoch<18 0.0001 20<0.00005<br>
Final model:5fold+SEResNext50*0.2+SEResNext50*0.8 <br>
Private Leader board 0.631<br>
