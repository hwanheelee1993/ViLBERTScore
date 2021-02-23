import torch
import pickle
import numpy as np
import os
from torch.utils.data import Dataset
import sys
import yaml

from easydict import EasyDict as edict
from pytorch_transformers.tokenization_bert import BertTokenizer
from vilbert.vilbert import VILBertForVLTasks, BertConfig, BertForMultiModalPreTraining
from vilbert.task_utils import LoadDatasetEval

import numpy as np
import matplotlib.pyplot as plt
import pdb
from math import log
from itertools import chain
from collections import defaultdict, Counter
from multiprocessing import Pool
from functools import partial
from copy import copy
tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased", do_lower_case=True
)

class CaptioningDataset(Dataset):
    def __init__(self, data_type='flickr8k', savedir='./', max_len=37, use_idf=False):
        super(CaptioningDataset, self).__init__()
        
        with open(os.path.join(savedir, 'cand_caps.pkl'), 'rb') as f:
            self.cand_caps = pickle.load(f, encoding='latin1')
        with open(os.path.join(savedir, 'gt_caps.pkl'), 'rb') as f:
            self.gt_caps = pickle.load(f, encoding='latin1')
        
        score_path = os.path.join(savedir, 'scores.pkl')
        if(os.path.exists(score_path)):
            with open(score_path, 'rb') as f:
                self.scores = pickle.load(f, encoding='latin1')
        else:
            self.scores = [1]*len(cand_canps) # without human scores
        with open(os.path.join(savedir, 'imgs_rcnn.pkl'), 'rb') as f:
            self.imgs = pickle.load(f, encoding='latin1')

        if(data_type=='pascal50s'):
            with open(os.path.join(savedir, 'ptypes.pkl'), 'rb') as f:
                self.ptypes = pickle.load(f, encoding='latin1')   

        else:
            with open(os.path.join(savedir, 'img_names.pkl'), 'rb') as f:
                self.img_names = pickle.load(f, encoding='latin1')      
                   
        self.data_type = data_type
        self.max_length = max_len
        self.use_idf = use_idf
        if(self.use_idf):
            with open(os.path.join(savedir, 'idf_dict.pkl'), 'rb') as f:
                self.idf_dict = pickle.load(f, encoding='latin1')

    '''
    def get_idf_dict(self, arr, tokenizer, nthreads=1):

        def process(a, tokenizer=None):
            if not tokenizer is None:
                a = ["[CLS]"]+tokenizer.tokenize(a)+["[SEP]"]
                a = tokenizer.convert_tokens_to_ids(a)
            return set(a)
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
    '''
    def tokenize(self, query):
        tokens = tokenizer.encode(query)
        tokens = tokenizer.add_special_tokens_single_sentence(tokens)
        #tokens = tokenizer.add_special_tokens(tokens)
        segment_ids = [0] * len(tokens)
        def get_idf(x):
            if x in self.idf_dict:
                return self.idf_dict[x]
            else:
                return max(self.idf_dict.values())
        input_mask = [1] * len(tokens)

        if(self.use_idf):        
            input_idf = [get_idf(x) for x in tokens]
        else:
            input_idf = copy(input_mask)

        if len(tokens) < self.max_length:
            # Note here we pad in front of the sentence
            padding = [0] * (self.max_length - len(tokens))
            tokens = tokens + padding
            input_mask += padding
            segment_ids += padding
            input_idf += padding
        else:
            tokens = tokens[:self.max_length]
            segment_ids = segment_ids[:self.max_length]
            input_mask = input_mask[:self.max_length]
            input_idf = input_idf[:self.max_length]

        text = torch.from_numpy(np.array(tokens))
        input_mask = torch.from_numpy(np.array(input_mask))
        input_idf = torch.from_numpy(np.array(input_idf))
        segment_ids = torch.from_numpy(np.array(segment_ids))
        return text, input_mask, segment_ids, input_idf

    def img_preprocess(self, img):
        image_w = img['image_width']
        image_h = img['image_height']
        feature = torch.from_numpy(img['features'])
        num_boxes = feature.size(0)
        
        g_feat = torch.sum(feature, dim=0) / num_boxes
        num_boxes = num_boxes + 1
        feature = torch.cat([g_feat.view(1,-1), feature], dim=0).float()
        boxes = img['bbox']
        image_location = np.zeros((boxes.shape[0], 5), dtype=np.float32)
        image_location[:,:4] = boxes
        image_location[:,4] = (image_location[:,3] - image_location[:,1]) * (image_location[:,2] - image_location[:,0]) / (float(image_w) * float(image_h))
        image_location[:,0] = image_location[:,0] / float(image_w)
        image_location[:,1] = image_location[:,1] / float(image_h)
        image_location[:,2] = image_location[:,2] / float(image_w)
        image_location[:,3] = image_location[:,3] / float(image_h)
        g_location = np.array([0,0,1,1,1])
        image_location = np.concatenate([np.expand_dims(g_location, axis=0), image_location], axis=0)
        image_mask = np.array([1] * (int(num_boxes))) 

        image_location = torch.tensor(image_location).float()
        image_mask = torch.tensor(image_mask).byte()
        co_attention_mask = torch.zeros((num_boxes, self.max_length))

        return feature, image_location, image_mask, co_attention_mask    

    def __getitem__(self, index):
        text_a = []
        input_mask_a = []
        segment_ids_a = []
        input_idf_a = []
        # Candidate Caption
        if(self.data_type == 'pascal50s'):
            for i in range(2):
                text_c, input_mask_c, segment_ids_c, input_idf = self.tokenize(self.cand_caps[index][i])
                text_a.append(text_c)
                input_mask_a.append(input_mask_c)
                segment_ids_a.append(segment_ids_c)
                input_idf_a.append(input_idf)
           
        else:
            text_c, input_mask_c, segment_ids_c, input_idf = self.tokenize(self.cand_caps[index].lower())
            text_a.append(text_c)
            input_mask_a.append(input_mask_c)
            segment_ids_a.append(segment_ids_c)
            input_idf_a.append(input_idf)

        # Reference Caption
        for i in range(len(self.gt_caps[index])):
            text_r, input_mask_r, segment_ids_r, input_idf = self.tokenize(self.gt_caps[index][i].lower())
            text_a.append(text_r)
            input_mask_a.append(input_mask_r)
            segment_ids_a.append(segment_ids_r)
            input_idf_a.append(input_idf)

        text_a = torch.stack(text_a, dim=0)
        input_mask_a = torch.stack(input_mask_a, dim=0)
        segment_ids_a = torch.stack(segment_ids_a, dim=0)    
        input_idf_a = torch.stack(input_idf_a, dim=0)
        features, spatials, image_mask, co_attention_mask = self.img_preprocess(self.imgs[index].item())

        return text_a, input_mask_a, segment_ids_a, features, spatials, image_mask, co_attention_mask, input_idf_a, index

    def __len__(self):
        return len(self.scores)            
    