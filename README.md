# XNOR-Net-Pytorch
This a PyTorch implementation of the [XNOR-Net](https://github.com/allenai/XNOR-Net). I implemented Binarized Neural Network (BNN) for:  
- Network-in-Network (NIN) for CIFAR-10
- AlexNet for ImageNet

## CIFAR-10
I implemented the NIN structure for the CIFAR-10 dataset. You can download the training and validation datasets [here](https://drive.google.com/open?id=0B-7I62GOSnZ8Z0ZCVXFtVnFEaTg) and uncompress the .zip file. To run the training:
```bash
$ cd <Repository Root>/CIFAR_10/
$ ln -s <Datasets Root> data
$ python main.py
```
Pretrained model can be downloaded [here](https://drive.google.com/open?id=0B-7I62GOSnZ8UjJqNnR1V0dMbWs). To evaluate the pretrained model:
```bash
$ cp <Pretrained Model> <Repository Root>/CIFAR_10/models/
$ python main.py --resume models/nin.best.pth.tar --evaluate
```

## ImageNet
I implemented the AlexNet for the ImageNet dataset. You can download the preprocessed dataset [here](https://drive.google.com/uc?export=download&id=0B-7I62GOSnZ8aENhOEtESVFHa2M) and uncompress it. However, to use this dataset, you have to install [Caffe](https://github.com/BVLC/caffe) first. Support with [torchvision](https://github.com/pytorch/vision) data reader will soon be added. If you need the function now, please contact ```jiecaoyu@umich.edu```.  
To set up the dataset:
```bash
$ cd <Repository Root>/ImageNet/networks/
$ ln -s <Datasets Root> data
```

### AlexNet
To train the network:
```bash
$ cd <Repository Root>/ImageNet/networks/
$ python main.py
```
Pretrained model can be downloaded [here](https://drive.google.com/open?id=0B-7I62GOSnZ8UjJqNnR1V0dMbWs). To evaluate the pretrained model:
```bash
$ cp <Pretrained Model> <Repository Root>/ImageNet/networks/
$ python main.py --resume alexnet.baseline.pth.tar --evaluate
```

## Todo
- Generate new dataset without caffe support.
- NIN for ImageNet.

## Notes
### Gradients of sign function
In the paper, the gradient in backward after the scaled sign function is  
  
![equation](http://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20C%7D%7B%5Cpartial%20W_i%7D%3D%5Cfrac%7B%5Cpartial%20C%7D%7B%5Cpartial%20%7B%5Cwidetilde%7BW%7D%7D_i%7D%20%28%5Cfrac%7B1%7D%7Bn%7D+%5Cfrac%7B%5Cpartial%20sign%28W_i%29%7D%7B%5Cpartial%20W_i%7D%5Ccdot%20%5Calpha%20%29)

<!--
\frac{\partial C}{\partial W_i}=\frac{\partial C}{\partial {\widetilde{W}}_i} (\frac{1}{n}+\frac{\partial sign(W_i)}{\partial W_i}\cdot \alpha )
-->
