import sys
import os
import torch
import yaml
import logging
logging.getLogger("pytorch_transformers").setLevel(logging.CRITICAL)
logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("vilbert.utils").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)

from easydict import EasyDict as edict
from pytorch_transformers.tokenization_bert import BertTokenizer
from vilbert.datasets import ConceptCapLoaderTrain, ConceptCapLoaderVal
from vilbert.vilbert import VILBertForVLTasks, BertConfig, BertForMultiModalPreTraining
from vilbert.task_utils import LoadDatasetEval
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL

from PIL import Image
import cv2
import argparse
import glob
from types import SimpleNamespace
import pdb

import pickle
import json
from tqdm import tqdm
from scipy.stats import kendalltau
from torch.nn.functional import softmax
from utils import *
from dataset import CaptioningDataset
from torch.utils.data import DataLoader


tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased', do_lower_case=True
)

args = SimpleNamespace(from_pretrained= "save/multi_task_model.bin",
                       bert_model="bert-base-uncased",
                       config_file="config/bert_base_6layer_6conect.json",
                       max_seq_length=101,
                       train_batch_size=1,
                       do_lower_case=True,
                       predict_feature=False,
                       seed=42,
                       num_workers=0,
                       baseline=False,
                       img_weight=1,
                       distributed=False,
                       objective=1,
                       visual_target=0,
                       dynamic_attention=False,
                       task_specific_tokens=True,
                       tasks='1',
                       save_name='',
                       in_memory=False,
                       batch_size=1,
                       local_rank=-1,
                       split='mteval',
                       clean_train_sets=True,
                       dataset='flickr8k',
                       task=7,
                       layer=-1,
                       expname='pretrain_cls_sep'
                      )
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="flickr8k")
# Use the task token "7" which is for "Retrieval COCO" in multi-task fine-tuned ViLBERT.
parser.add_argument("--task", type=int, default=7)
parser.add_argument("--layer", type=int, default=-1)
parser.add_argument("--datadir", type=str, default='data')
parser.add_argument("--model_path",type=str, default='save/multi_task_model.bin')
parser.add_argument("--compute_correlation",type=bool, default=False)

args_ = parser.parse_args()
args.dataset = args_.dataset
args.datadir = args_.datadir
args.task = args_.task
args.layer = args_.layer
args.from_pretrained = args_.model_path
args.compute_correlation = args_.compute_correlation

if(args.from_pretrained == 'save/pretrained_model.bin'):
    args.task_specific_tokens = False

config = BertConfig.from_json_file(args.config_file)
with open('./vilbert_tasks.yml', 'r') as f:
    task_cfg = edict(yaml.safe_load(f))

task_names = []
for i, task_id in enumerate(args.tasks.split('-')):
    task = 'TASK' + task_id
    name = task_cfg[task]['name']
    task_names.append(name)

timeStamp = args.from_pretrained.split('/')[-1] + '-' + args.save_name
config = BertConfig.from_json_file(args.config_file)
default_gpu=True

if args.predict_feature:
    config.v_target_size = 2048
    config.predict_feature = True
else:
    config.v_target_size = 1601
    config.predict_feature = False

if args.task_specific_tokens:
    config.task_specific_tokens = True    

if args.dynamic_attention:
    config.dynamic_attention = True

config.visualization = True
num_labels = 3129

if args.baseline:
    model = BaseBertForVLTasks.from_pretrained(
        args.from_pretrained, config=config, num_labels=num_labels, default_gpu=default_gpu
        )
else:
    model = VILBertForVLTasks.from_pretrained(
        args.from_pretrained, config=config, num_labels=num_labels, default_gpu=default_gpu
        )
    
model.eval()
cuda = torch.cuda.is_available()
if cuda: model = model.cuda(0)
tokenizer = BertTokenizer.from_pretrained(
    args.bert_model, do_lower_case=args.do_lower_case
)

datadir = args.datadir
data_type = args.dataset
savedir = os.path.join(datadir, data_type)

from math import log
from itertools import chain
from collections import defaultdict, Counter
from multiprocessing import Pool
from functools import partial

def process(a, tokenizer=None):
    if not tokenizer is None:
        a = ["[CLS]"]+tokenizer.tokenize(a)+["[SEP]"]
        a = tokenizer.convert_tokens_to_ids(a)
    return set(a)

def get_idf_dict(arr, tokenizer, nthreads=1):

    """
    Returns mapping from word piece index to its inverse document frequency.
    Args:
        - :param: `arr` (list of str) : sentences to process.
        - :param: `tokenizer` : a BERT tokenizer corresponds to `model`.
        - :param: `nthreads` (int) : number of CPU threads to use
    """
    idf_count = Counter()
    num_docs = len(arr)

    process_partial = partial(process, tokenizer=tokenizer)

    with Pool(nthreads) as p:
        idf_count.update(chain.from_iterable(p.map(process_partial, arr)))

    idf_dict = defaultdict(lambda : log((num_docs+1)/(1)))
    idf_dict.update({idx:log((num_docs+1)/(c+1)) for (idx, c) in idf_count.items()})
    return idf_dict

dataset = CaptioningDataset(data_type=data_type, savedir=savedir, use_idf=False)
dataloader = DataLoader(dataset, 64, shuffle=False)

def compute_bert_score(model, text_a, input_mask_a, segment_ids_a, 
                       features, spatials, image_mask, co_attention_mask, input_idf_a, task, x, y, 
                       use_idf=False, layer=-1):
    with torch.no_grad():
        p5 = []
        r5 = []
        f5 = []            

        st_c, sv_c, pt_c, pv_c, att_c = model.bert(
            text_a[:,x,:], features, spatials, segment_ids_a[:,x,:],
            input_mask_a[:,x,:], image_mask, co_attention_mask, task,
        output_all_encoded_layers=True)
        
        for i in range(5):
            st_g, sv_g, pt_g, pv_g, att_g = model.bert(
                text_a[:,y+i,:], features, spatials, segment_ids_a[:,y+i,:],
                input_mask_a[:,y+i,:], image_mask, co_attention_mask, task,
            output_all_encoded_layers=True)
            if(use_idf):
                p, r, f = bert_score(st_g[layer], st_c[layer], input_idf_a[:,y+i, :], input_idf_a[:,x, :])
            else:
                p, r, f = bert_score(st_g[layer], st_c[layer], input_mask_a[:,y+i, :], input_mask_a[:,x, :])
            p5.append(p)
            r5.append(r)
            f5.append(f)        

    p5a = np.average(p5, axis=0)
    r5a = np.average(r5, axis=0)
    f5a = np.average(f5, axis=0) 
    
    return p5a, r5a, f5a

layer = args.layer

prs_a = []
rcs_a = []
f1s_a = []

use_idf = False

#for idx in tqdm(range(len(scores))):
for text_a, input_mask_a, segment_ids_a, features, spatials, image_mask, co_attention_mask, input_idf_a, idxs_ in tqdm(iter(dataloader)):
    text_a = text_a.cuda() 
    input_idf_a = input_idf_a.cuda()
    input_mask_a = input_mask_a.cuda()
    segment_ids_a = segment_ids_a.cuda()
    features = features.cuda()
    spatials = spatials.cuda()
    image_mask = image_mask.cuda()
    co_attention_mask = co_attention_mask.cuda()
    task = [args.task]
    task = torch.from_numpy(np.array(task)).cuda().unsqueeze(0).repeat(spatials.size(0), 1)
    #break
    with torch.no_grad():
        p5a, r5a, f5a = compute_bert_score(model, text_a, input_mask_a, segment_ids_a, 
                                                           features, spatials, image_mask, co_attention_mask, input_idf_a, task, 0, 1, layer=layer)
        if(len(prs_a) == 0):
            prs_a = p5a
            rcs_a = r5a
            f1s_a = f5a            
        else:
            prs_a = np.concatenate((prs_a, p5a))
            rcs_a = np.concatenate((rcs_a, r5a))
            f1s_a = np.concatenate((f1s_a, f5a))

final_results = [prs_a, rcs_a, f1s_a]


if(args.compute_correlation):
    scores = dataset.scores
    print("Kendall Correlation Coefficient")
    print("P: %.3f"%kendalltau(scores, prs_a)[0])
    print("R: %.3f"%kendalltau(scores, rcs_a)[0])
    print("F: %.3f"%kendalltau(scores, f1s_a)[0])

# Save the results
savefile = 'results/'+args.dataset+'.pkl'
print('Saved the results to %s'%savefile)
with open(savefile, 'wb') as f:
    pickle.dump(final_results, f)