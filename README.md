# Flatten Anything: Unsupervised Neural Surface Parameterization (NeurIPS-2024)



This is the official implementation of **[[Flatten Anything Model (FAM)](https://arxiv.org/abs/2405.14633)]**, an unsupervised neural architecture to achieve global free-boundary surface parameterization via learning point-wise mappings between 3D points on the target geometric surface and adaptively-deformed UV coordinates within the 2D parameter domain. This code has been tested with Python 3.9, PyTorch 1.10.1, CUDA 11.1 and cuDNN 8.0.5 on Ubuntu 20.04.

<p align="center"> <img src="https://github.com/keeganhk/FlattenAnything/blob/master/imgs/examples.png" width="65%"> </p>

<p align="center"> <img src="https://github.com/keeganhk/FlattenAnything/blob/master/imgs/workflow.png" width="65%"> </p>

### Instruction

- Within ```cdbs/CD/```, run ```python setup.py install``` for compilation.

- Within ```cdbs/EMD/```, run ```python setup.py install``` and ```cp build/lib.linux-x86_64-cpython-39/emd_cuda.cpython-39-x86_64-linux-gnu.so .``` for compilation.

- A demo script for running FAM is provided in a jupyter notebook ```scripts/main.ipynb```.
