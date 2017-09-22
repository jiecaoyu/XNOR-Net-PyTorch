# XNOR-Net-Pytorch
This a PyTorch implementation of the [XNOR-Net](https://github.com/allenai/XNOR-Net). I implemented Binarized Neural Network (BNN) for:  
- Network-in-Network (NIN) for CIFAR-10
- AlexNet for ImageNet
## CIFAR-10
I implemented the NIN structure for the CIFAR-10 dataset. You can download the training and validation datasets [here](https://drive.google.com/open?id=0B-7I62GOSnZ8Z0ZCVXFtVnFEaTg) and uncompress the .zip file. To run the training:
```bash
$ cd <Repository Root>/CIFAR_10/
$ ln -s data <Datasets Root>
$ python main.py
```
Pretrained model can be downloaded [here](https://drive.google.com/open?id=0B-7I62GOSnZ8UjJqNnR1V0dMbWs).

## Notes
### Gradients of sign function
In the paper, the gradient in backward after the scaled sign function is  
  
![equation](http://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20C%7D%7B%5Cpartial%20W_i%7D%3D%5Cfrac%7B%5Cpartial%20C%7D%7B%5Cpartial%20%7B%5Cwidetilde%7BW%7D%7D_i%7D%20%28%5Cfrac%7B1%7D%7Bn%7D+%5Cfrac%7B%5Cpartial%20sign%28W_i%29%7D%7B%5Cpartial%20W_i%7D%5Ccdot%20%5Calpha%20%29)

<!--
\frac{\partial C}{\partial W_i}=\frac{\partial C}{\partial {\widetilde{W}}_i} (\frac{1}{n}+\frac{\partial sign(W_i)}{\partial W_i}\cdot \alpha )
-->
