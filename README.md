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

## Miscellaneous

Code and data under `util/` and `data/referit/` are borrowed from [text_objseg](https://github.com/ronghanghu/text_objseg) and slightly modified for compatibility with Tensorflow 1.2.1.

## TODO

Add TensorBoard support.
