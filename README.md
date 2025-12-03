# CompNet

CompNet: Competitive Neural Network for Palmprint Recognition Using Learnable Gabor Kernels

[ [paper](https://ieeexplore.ieee.org/document/9512475) | [supp](https://github.com/xuliangcs/compnet/blob/main/Supplementary%20Material.pdf) | [cite](./res/cite.txt) | [license](./LICENSE) ]

<img src="./res/compnet.png" alt="Framework of the CompNet" style="zoom:80%;" />

## 1. Related Materials

- Paper: [online](https://ieeexplore.ieee.org/document/9512475)

- Supplementary Material: [pdf](https://github.com/xuliangcs/compnet/blob/main/Supplementary%20Material.pdf)

- Pretrained Models: [google](https://drive.google.com/drive/folders/1TuqQVI0T9pBVr2jQKLY40jbZ8ZMQLfie?usp=sharing) or [baidu](https://pan.baidu.com/s/1BEOylWlj1MlPb5pfw57fKw?pwd=dl4m)

- Publicly Available Datasets: [Tongji](https://cslinzhang.github.io/ContactlessPalm), [IITD](https://www4.comp.polyu.edu.hk/~csajaykr/IITD/Database_Palm.htm), [REST](https://ieee-dataport.org/open-access/rest-database), [NTU](https://github.com/BFLTeam/NTU_Dataset), [XJTU-UP](https://gr.xjtu.edu.cn/en/web/bell)

## 2. Visualizations
![](./res/05_cb2_gabor.gif)

Fig. 2.1 **CB17x17_Gabor** feature maps obtained at different epochs

![](./res/06_cb2_scc.gif)

Fig. 2.2 **CB17x17_SCC** feature maps obtained at different epochs

![](./res/11_cb3_conv1.gif)

Fig. 2.3 **CB7x7_Conv1** feature maps obtained at different epochs 

<img src="./res/08_cb2_conv2.gif" style="zoom:200%;" />

Fig. 2.4 **CB17x17_Conv2** feature maps obtained at different epochs

> Each row represents feature maps obtained from one ROI image, and each column corresponds to a single feature channel.



# 3. Requirements
![](https://img.shields.io/badge/Ubuntu-tested-green) ![](https://img.shields.io/badge/Windows11-tested-green) 

Recommanded hardware requirement **for training**:
- GPU Mem $\gt$ 6G
- CPU Mem $\geq$ 16G (32G is recommended for highspeed data augmentation)

Software development environment:
- [cuda&cudnn&gpu-driver](https://github.com/xuliangcs/env/blob/main/doc/PyTorch.md)
- `Anaconda`: [download & install](https://www.anaconda.com/download/success)
- `PyTorch`: installation command lines are as follows
  ```
  conda create -n compnet python=3.8 
  conda activate compnet

  conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch

  pip install -r requirements.txt
  ```
tips:
- `requirements.txt` could be found at the root folder of this project
- for PyTorch 1.7 with **other CUDA versions**, please refer to the official [pytorch installation commands](https://pytorch.org/get-started/previous-versions/#v170)
- the speed of different network servers and conda sources varies a lot when install the above packages
- for more details of the software environment, pls refer to the `pip list` [rst](./res/piplist.md)
- version selection strategy: [pytorch](https://pytorch.org/get-started/previous-versions/)->[cuda](https://developer.nvidia.com/cuda-toolkit-archive)->[cudnn](https://developer.nvidia.com/cudnn-downloads)


Other tested versions:
> - cuda: 10.2 / 11.0
> - cudnn: 7.6.5.32 / 9.1.0.70
> - pytorch: 1.2 / 1.7
> - torchvision: 0.4 / 0.8
> - python: 3.7.4 / 3.8.19
> - opencv: 3.2.7 / 4.8.1.78
> - numpy: 1.16.4 / 1.24.3
> - scikit-learn: 0.21.3 / 1.0.2
> - scipy: 1.3.1 / 1.10.1

Tips: If a particular version is no longer available for download, you can try replacing it with a newer version. However, running the code may encounter compatibility issues, such as deprecated functions or changes in default parameters. Please search for solutions based on the error messages and the actual intent of the code.


## 4. PyTorch Implementation

**Configurations**

1. modify `path1` and `path2` in `genText.py`

    - `path1`: path of the training set (e.g., Tongji session1)
    - `path2`: path of the testing set (e.g., Tongji session2)
    
2. modify `num_classes` in `train.py`, `test.py`, and `inference.py`
    - Tongji: 600, IITD: 460, REST: 358, XJTU-UP: 200, KTU: 145

**Dataset preparation**


**Commands**

```shell
cd path/to/CompNet/
#in the CompNet folder:

#prepare data
cp ./data/for_reference/genText_xxx.py ./data/genText.py
#where xxx is the dataset name, e.g., tongji =>genText_tongji.py
Modify the DB path variable in ./data/genText.py
#the sample naming format should be consistent with the script's requirements

#generate the training and testing data sets
python ./data/genText.py
mv ./train.txt ./data/
mv ./test.txt ./data/

#train the network
python train.py
#test the model
python test.py
#inference
python inference.py

#Metrics
#obtain the genuine-impostor matching score distribution curve
python    getGI.py   ./rst/veriEER/scores_xxx.txt    scores_xxx
#obtain the EER and the ROC curve
python    getEER.py   ./rst/veriEER/scores_xxx.txt    scores_xxx
```
The `.pth` file will be generated at the current folder, and all the other results will be generated in the `./rst` folder.


🗞️tips: GPU -> CPU ([more details](https://pytorch.org/docs/2.1/generated/torch.load.html)):
```bash
# training on GPU, test on CPU
torch.load('net_params.pth', map_location='cpu')
```


## 5. Framework

```shell
compnet(
  (cb1): CompetitiveBlock(
    (gabor_conv2d): GaborConv2d()
    (argmax): Softmax(dim=1)
    (conv1): Conv2d(9, 32, kernel_size=(5, 5), stride=(1, 1))
    (maxpool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (conv2): Conv2d(32, 12, kernel_size=(1, 1), stride=(1, 1))
  )
  (cb2): CompetitiveBlock(
    (gabor_conv2d): GaborConv2d()
    (argmax): Softmax(dim=1)
    (conv1): Conv2d(9, 32, kernel_size=(5, 5), stride=(1, 1))
    (maxpool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (conv2): Conv2d(32, 12, kernel_size=(1, 1), stride=(1, 1))
  )
  (cb3): CompetitiveBlock(
    (gabor_conv2d): GaborConv2d()
    (argmax): Softmax(dim=1)
    (conv1): Conv2d(9, 32, kernel_size=(5, 5), stride=(1, 1))
    (maxpool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (conv2): Conv2d(32, 12, kernel_size=(1, 1), stride=(1, 1))
  )
  (fc): Linear(in_features=9708, out_features=512, bias=True)
  (drop): Dropout(p=0.25, inplace=False)
  (arclayer): ArcMarginProduct()
)
```

## 6. Citation
🌻If it helps you, we would like you to cite the following paper:🌱

```tex
@article{spl2021compnet,
author={Liang, Xu and Yang, Jinyang and Lu, Guangming and Zhang, David},
journal={IEEE Signal Processing Letters},
title={CompNet: Competitive Neural Network for Palmprint Recognition Using Learnable Gabor Kernels},
year={2021},
volume={28},
number={},
pages={1739-1743},
doi={10.1109/LSP.2021.3103475}}
```

Xu Liang, Jinyang Yang, Guangming Lu and David Zhang, "CompNet: Competitive Neural Network for Palmprint Recognition Using Learnable Gabor Kernels," in IEEE Signal Processing Letters, vol. 28, pp. 1739-1743, 2021, doi: 10.1109/LSP.2021.3103475.

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

## Acknowledgement

Portions of the research use the REST'2016 Database collected by the Research Groups in Intelligent Machines, University of Sfax, Tunisia. We would also like to thank the organizers (IITD, Tongji, REgim, XJTU, and NTU) for allowing us to use their datasets. 


## References

[1] J. Deng, J. Guo,  N. Xue  and  S.  Zafeiriou,  “ArcFace:  Additive  angularmargin  loss  for  deep  face  recognition,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4690–4699, Jun. 2019.

[2] Arcface PyTorch Implementation  (`ArcMarginProduct`). [Online]. Available: [arcface](https://github.com/ronghuaiyang/arcface-pytorch)

[3] P. Chen, W. Li, L. Sun, X. Ning, L. Yu, and L. Zhang, “LGCN: LearnableGabor convolution network for human gender recognition in the wild,” IEICE Transactions on Information and Systems, 102(10), pp. 2067–2071, Oct. 2019.

[4] A. Genovese, V. Piuri, K. N. Plataniotis and F. Scotti, “PalmNet: Gabor-PCA convolutional networks for touchless palmprint recognition,” IEEE Transactions on Information Forensics and Security, 14(12), pp. 3160–3174, Dec. 2019. [palmnet](https://github.com/AngeloUNIMI/PalmNet)

[5] X. Liang, D. Fan, J. Yang, W. Jia, G. Lu and D. Zhang, "PKLNet: Keypoint Localization Neural Network for Touchless Palmprint Recognition Based on Edge-Aware Regression," in IEEE Journal of Selected Topics in Signal Processing, 17(3), pp. 662-676, May 2023, [doi](https://ieeexplore.ieee.org/document/10049596): 10.1109/JSTSP.2023.3241540. (`Palmprint ROI extraction`) [pklnet](https://github.com/xuliangcs/pklnet)🖖

[6] X. Liang, Z. Li, D. Fan, B. Zhang, G. Lu and D. Zhang, "Innovative Contactless Palmprint Recognition System Based on Dual-Camera Alignment," in IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 52, no. 10, pp. 6464-6476, Oct. 2022, [doi](https://ieeexplore.ieee.org/document/9707646): 10.1109/TSMC.2022.3146777. (`Bimodal alignment`) [ppnet](https://github.com/xuliangcs/ppnet)🖐️

[7] PyTorch API Documents: https://pytorch.org/docs/stable/index.html