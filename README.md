# Recurrent Multimodal Interaction for Referring Image Segmentation

This repository contains code for [Recurrent Multimodal Interaction for Referring Image Segmentation](https://arxiv.org/abs/1703.07939), ICCV 2017.

If you use the code, please cite
```
@inproceedings{liu2017recurrent,
  title={Recurrent Multimodal Interaction for Referring Image Segmentation},
  author={Liu, Chenxi and Lin, Zhe and Shen, Xiaohui and Yang, Jimei and Lu, Xin and Yuille, Alan},
  booktitle={{ICCV}},
  year={2017}
}
```

## Setup

- Tensorflow 1.2.1
- [Download](http://mscoco.org/dataset/#download) or use symlink, such that the MS COCO images are under `data/coco/images/train2014/`
- [Download](http://www.eecs.berkeley.edu/~ronghang/projects/cvpr16_text_obj_retrieval/referitdata.tar.gz) or use symlink, such that the ReferItGame data are under `data/referit/images` and `data/referit/mask`
- Run `mkdir external`. Download, git clone, or use symlink, such that [TF-resnet](https://github.com/chenxi116/TF-resnet) and [TF-deeplab](https://github.com/chenxi116/TF-deeplab) are under `external`. Then strictly follow the `Example Usage` section of their README
- Download, git clone, or use symlink, such that [refer](https://github.com/chenxi116/refer) is under `external`. Then strictly follow the `Setup` and `Download` section of its README. Also put the `refer` folder in `PYTHONPATH`
- Download, git clone, or use symlink, such that the [MS COCO API](https://github.com/pdollar/coco) is under `external` (i.e. `external/coco/PythonAPI/pycocotools`)
- [pydensecrf](https://github.com/lucasb-eyer/pydensecrf)

## Data Preparation

```
python build_batches.py -d Gref -t train
python build_batches.py -d Gref -t val
python build_batches.py -d unc -t train
python build_batches.py -d unc -t val
python build_batches.py -d unc -t testA
python build_batches.py -d unc -t testB
python build_batches.py -d unc+ -t train
python build_batches.py -d unc+ -t val
python build_batches.py -d unc+ -t testA
python build_batches.py -d unc+ -t testB
python build_batches.py -d referit -t trainval
python build_batches.py -d referit -t test
```

## Training and Testing

Specify several options/flags and then run `main.py`:

- `-g`: Which GPU to use. Default is 0.
- `-m`: `train` or `test`. Training mode or testing mode.
- `-w`: `resnet` or `deeplab`. Specify pre-trained weights.
- `-n`: `LSTM` or `RMI`. Model name.
- `-d`: `Gref` or `unc` or `unc+` or `referit`. Specify dataset.
- `-t`: `train` or `trainval` or `val` or `test` or `testA` or `testB`. Which set to train/test on.
- `-i`: Number of training iterations in training mode. The iteration number of a snapshot in testing mode. 
- `-s`: Used only in training mode. How many iterations per snapshot.
- `-v`: Used only in testing mode. Whether to visualize the prediction. Default is False.
- `-c`: Used only in testing mode. Whether to also apply Dense CRF. Default is False.

For example, to train the ResNet + LSTM model on Google-Ref using GPU 2, run
```
python main.py -m train -w resnet -n LSTM -d Gref -t train -g 2 -i 750000 -s 50000
```
To test the 650000-iteration snapshot of the DeepLab + RMI model on UNC testA set using GPU 1 (with visualization and Dense CRF), run
```
python main.py -m test -w deeplab -n RMI -d unc -t testA -g 1 -i 650000 -v -c
```

## Miscellaneous

Code and data under `util/` and `data/referit/` are borrowed from [text_objseg](https://github.com/ronghanghu/text_objseg) and slightly modified for compatibility with Tensorflow 1.2.1.

## TODO

Add TensorBoard support.
