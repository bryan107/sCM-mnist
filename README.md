# sCM-mnist
#### [Paper](https://arxiv.org/abs/2410.11081)
Unofficial implementation of "Simplifying, Stabilizing & Scaling Continuous-Time Consistency Models" for MNIST

![The Algorithm](https://github.com/user-attachments/assets/a09d384c-e353-4466-8317-041a96d9d536)

This is a simple 2-file implementation of consistency training for MNIST. It can be easily adapted to consistency distillation in a few lines.
The code is intended to be short, and extremely easy to read and adapt to other use cases. unet.py contains a UNet for MNIST, for which I am borrowing from [MNISTDiffusion](https://github.com/bot66/MNISTDiffusion). 
train.py contains the code for the consistency training algorithm, with many comments to explain details that aren't covered in the algorithm description above.
