# sCM-mnist
#### [Paper](https://arxiv.org/abs/2410.11081)
Unofficial implementation of "Simplifying, Stabilizing & Scaling Continuous-Time Consistency Models" for MNIST

![The Algorithm](https://github.com/user-attachments/assets/a09d384c-e353-4466-8317-041a96d9d536)

This is a simple implementation of consistency training for MNIST.
The code is intended to be short, and extremely easy to read and adapt to other use cases:

### train_consistency.py

Contains the core algorithm from the paper. The script will do consistency training by default, but if a 'model.pt' file exists, it will use it to do consistency distillation. 'model.pt' can be created with train_diffusion.py.

train.py uses torch.func.jvp, which calculates the JVP in the forward pass. Only some operations support JVP, but most common operations are supported. 
There are two exceptions: BatchNorm and Flash Attention. BatchNorm has some in-place operations that cause problems. The simple solution to this is to just avoid BatchNorm. I use GroupNorm in this implementation. Flash attention is more complicated, but the paper provides a way to calculate the JVP efficiently. I may implement this here in the future. Note that normal attention (see [here](https://github.com/NVlabs/edm/blob/main/training/networks.py) for one implementation) supports JVP.

**Training Tip**: The paper claims that a prior weighting of `1 / sigma` in the loss lowers variance, but in many of my experiments I found that this destabilizes training. If you experience similar issues, replacing `1 / sigma` with just `1` may help.

### train_diffusion.py

This is a simple script to train a diffusion model with the TrigFlow parameterization. It creates 'model.pt' which can be distilled with train_consistency.py.

### unet.py

This is a simple UNet for MNIST. I ended up borrowing the UNet from [MNISTDiffusion](https://github.com/bot66/MNISTDiffusion).
Generally, the specifics of the model are not that important. However, many models may face issues with unstable gradients from time embeddings (see paper for details).
There are a few changes that are necessary to ensure that training is stable:
- Using Positional Embeddings instead of learned or fourier embeddings (Fourier embeddings with a small fourier scale can still work well)
- Using AdaGN *with PixelNorm* in order to inject time embeddings
- Added logvar output, which is used for adaptive loss weights
  
These changes are usually all that is necessary to ensure that training is stable for most models. 

#### Toy

There is also a folder called toy which contains some scripts for training a toy model. The script also outputs a plot of the model's gradients during training, which helps to visualize how consistency models work.

## Results

Diffusion model (100 steps)

![epoch_0020](https://github.com/user-attachments/assets/bc3f8b54-c55d-452b-a59c-bc1b0e02c2e7)


Consistency model (1 step)

![epoch_0020](https://github.com/user-attachments/assets/4fcb78d0-c2f9-47e8-aa2a-ef8b1f173c06)

Consistency model (2 step)

![epoch_0020](https://github.com/user-attachments/assets/e13ed5d2-0c7b-463d-b267-976ea2ea235f)
