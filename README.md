## Is Learning Always Necessary for Solving Fourier Phase Retrieval with a Reference Image?
This repository containes code for our submission to the NeurIPS 2021 Workshop on Deep Learning and Inverse Problems.

## Contents
```
|-- references
|   |-- gs
|   |   |-- non-oversampled
|   |   |   |-- u_cifar_gs.npy
|   |   |   |-- u_emnist_gs.npy
|   |   |   |-- u_fmnist_gs.npy
|   |   |   |-- u_mnist_gs.npy
|   |   |   `-- u_svhn_gs.npy
|   |   `-- oversampled
|   |       |-- u_cifar.npy
|   |       |-- u_emnist.npy
|   |       |-- u_fmnist.npy
|   |       |-- u_mnist.npy
|   |       `-- u_svhn.npy
|   |-- hyder
|   |   |-- non-oversampled
|   |   |   |-- u_cifar.npy
|   |   |   |-- u_emnist.npy
|   |   |   |-- u_fmnist.npy
|   |   |   |-- u_mnist.npy
|   |   |   `-- u_svhn.npy
|   |   `-- oversampled
|   |       |-- u_celeba.npy
|   |       |-- u_cifar.npy
|   |       |-- u_emnist.npy
|   |       |-- u_fmnist.npy
|   |       |-- u_mnist.npy
|   |       `-- u_svhn.npy
|   `-- random
|       |-- u_ours_noiseless.npy
|       |-- u_ours.npy
|       |-- u_random_binary.npy
|       `-- u_random.npy
|-- data.py
|-- phase-retrieval-with-reference.ipynb
|-- README.md
|-- unrolled-GS.ipynb
`-- util.py
    
```

### Requirements
All experiments were conducted with the following package versions:
- numpy==1.19.5
- torch==1.9.0
- torchvision==0.10.0
- matplotlib==3.4.3
- scikit-image==0.17.2

The reference images for the oversampled case dicussed in Hyder et al. [1] were obtained from the [official repository](https://github.com/CSIPlab/learn-reference-pr).

### References
[1] Rakib Hyder, Zikui Cai, and M Salman Asif. Solving phase retrieval with a learned reference. In European Conference on Computer Vision, pages 425–441. Springer, 2020.
