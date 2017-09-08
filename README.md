# XNOR-Net-Pytorch
This a PyTorch implementation of the [XNOR-Net](https://github.com/allenai/XNOR-Net).

## Notes
### Gradients of sign function
In the paper, the gradient in backward after the scaled sign function is  
  
![equation](http://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20C%7D%7B%5Cpartial%20W_i%7D%3D%5Cfrac%7B%5Cpartial%20C%7D%7B%5Cpartial%20%7B%5Cwidetilde%7BW%7D%7D_i%7D%20%28%5Cfrac%7B1%7D%7Bn%7D+%5Cfrac%7B%5Cpartial%20sign%28W_i%29%7D%7B%5Cpartial%20W_i%7D%5Ccdot%20%5Calpha%20%29)

<!--
\frac{\partial C}{\partial W_i}=\frac{\partial C}{\partial {\widetilde{W}}_i} (\frac{1}{n}+\frac{\partial sign(W_i)}{\partial W_i}\cdot \alpha )
-->
