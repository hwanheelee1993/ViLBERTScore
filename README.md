# ViLBERTScore

This repository provides an evaluation metric for image captioning using ViLBERT which is based on our paper [ViLBERTScore: Evaluating Image Caption Using Vision-and-Language BERT](https://www.aclweb.org/anthology/2020.eval4nlp-1.4/).

## Repository Setup

This code is built upon original [ViLBERT](https://arxiv.org/abs/1908.02265) paper and its [repository](https://github.com/facebookresearch/vilbert-multi-task). We provide the almost same guideline as in the original repository as follows.

1. Create a fresh conda environment, and install all dependencies.

```text
conda create -n vilbert-score python=3.6
conda activate vilbert-score
git clone https://github.com/hwanheelee1993/ViLBERTScore.git
cd ViLBERTScore
pip install -r requirements.txt
```

2. Install pytorch
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

3. Install this codebase as a package in this environment.
```text
python setup.py develop
```

## Pre-trained models

We used two pre-trained models(pretrained ViLBERT, fine-tuned on 12 tasks) in our work. Please download the models in original [ViLBERT repository](https://github.com/facebookresearch/vilbert-multi-task) and save it to "save" dir. 
(two files: "pretrained_model.bin", "multi_task_model.bin")

## Feature Extraction

### Using Processed Dataset
We provide the processed version for Flickr8k and Composite in [this link](http://milabfile.snu.ac.kr:15000/sharing/VEdaKoEwz), including the pre-computed detection features. Download the files and extract to "data" dir.

### Extracting Features for Other Dataset
We extract the detection features following the guidelines in [this link](https://github.com/facebookresearch/vilbert-multi-task/tree/master/data).
Please extract the features from the link and save them as "imgs_rcnn.pkl" which is a list of each feature. 

Then make files, "cand_caps.pkl", "gt_caps.pkl",("scores.pkl" optional for computing correlation) which are the list of each feature. 

## Computing Score
You can compute the scores using the following code.
```text
python compute_vilbertscore.py --dataset flickr8k
```

## License
MIT license

Please cite the following paper if you use this code. 
[ViLBERTScore: Evaluating Image Caption Using Vision-and-Language BERT](https://www.aclweb.org/anthology/2020.eval4nlp-1.4/).
```
@inproceedings{lee2020vilbertscore,
  title={ViLBERTScore: Evaluating Image Caption Using Vision-and-Language BERT},
  author={Lee, Hwanhee and Yoon, Seunghyun and Dernoncourt, Franck and Kim, Doo Soon and Bui, Trung and Jung, Kyomin},
  booktitle={Proceedings of the First Workshop on Evaluation and Comparison of NLP Systems},
  pages={34--39},
  year={2020}
}
```
