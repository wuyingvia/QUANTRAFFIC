# QUANTRAFFIC
Adaptive Modeling of Uncertainties for Traffic Forecasting

[The preprint version has been uploaded to arXiv. ](https://arxiv.org/pdf/2303.09273.pdf)

# Cite

​```
@ARTICLE{10304591,
  author={Wu, Ying and Ye, Yongchao and Zeb, Adnan and Yu, James Jianqiao and Wang, Zheng},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={Adaptive Modeling of Uncertainties for Traffic Forecasting}, 
  year={2023},
  pages={1-16},
  doi={10.1109/TITS.2023.3327100}}

​```

## Introduction

QUANTRAFFIC is a generic framework for quantifying the prediction uncertainties of DNN-based traffic forecasting models. Given a certain confidence level, the QUANTRAFFIC
framework computes a PI that defines a range of possible values within which the real data (e.g., traffic time, flow, or speed) will likely fall.

## Installation Dependencies
- Ubuntu 18.04.5
- Python 3 (>= 3.6)
- PyTorch version 1.8.0.
- Pandas

## Public data
- METR-LA
- PeMS-BAY
- PeMSD7(M)
- PeMS03
- PeMS04
- PeMS07
- PeMS08

For METR-LA, and PeMS-BAY, thanks to [DCRNN](https://github.com/liyaguang/DCRNN).

For PeMSD7(M), thanks to [STGCN](https://github.com/VeritasYin/STGCN_IJCAI-18).

For PeMS03/04/07/08, thanks to [ASTGNN](https://github.com/guoshnBJTU/ASTGNN)

## Basic model
- GraphWaveNet

## note
我知道这个代码很屎，没事，可以喷的~..【泪目

所以算了，屎山虽然屎，但好歹能跑【撒花

新版本应该会在春节前后更新【拖延本上第一条todo 

有啥想法请邮箱联系wuyingvia@outlook.com, 欢迎交流～
