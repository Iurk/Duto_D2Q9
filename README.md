# LBM
Lattice-Boltzmann -> Cylinder case

Código de simulação utilizando o método LBM (Lattice-Boltzmann), com paralelização por GPU.
O problema simulado é o caso do cilindro.

Para o correto funcionamento do código são necessários os seguintes pacotes:

* CUDA drivers - Drivers para a paralelização utilizando a GPU. Disponível no site da [Nvidia](https://developer.nvidia.com/cuda-downloads) e o guia de instalação disponível no [aqui](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)

* yaml-cpp - Repositório para a leitura de arquivos .yml em C++. Disponível [nesse link](https://github.com/jbeder/yaml-cpp)

* PyYAML - Biblioteca para leitura de arquivos .yml em Python. Para instalar, basta utilizar o pip, seguindo o comando

```
pip install pip install PyYAML
```

* imageio-ffmpeg - Biblioteca para a geração da animação em .mp4 em Python. Para instalar, basta utilizar o pip, seguindo o comando

```
pip install pip install imageio-ffmpeg
```
