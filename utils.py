import sys
import os
import torch
import yaml

from easydict import EasyDict as edict
from pytorch_transformers.tokenization_bert import BertTokenizer
from vilbert.datasets import ConceptCapLoaderTrain, ConceptCapLoaderVal
from vilbert.vilbert import VILBertForVLTasks, BertConfig, BertForMultiModalPreTraining
from vilbert.task_utils import LoadDatasetEval

import numpy as np
import matplotlib.pyplot as plt
import PIL

# maskrcnn_benchmark.config import cfg
#from maskrcnn_benchmark.layers import nms
#from maskrcnn_benchmark.modeling.detector import build_detection_model
#from maskrcnn_benchmark.structures.image_list import to_image_list
#from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from PIL import Image
import cv2
import argparse
import glob
from types import SimpleNamespace
import pdb

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased", do_lower_case=True
)

def tokenize_batch(batch):
    return [tokenizer.convert_tokens_to_ids(sent) for sent in batch]

def untokenize_batch(batch):
    return [tokenizer.convert_ids_to_tokens(sent) for sent in batch]

def detokenize(sent):
    """ Roughly detokenizes (mainly undoes wordpiece) """
    new_sent = []
    for i, tok in enumerate(sent):
        if tok.startswith("##"):
            new_sent[len(new_sent) - 1] = new_sent[len(new_sent) - 1] + tok[2:]
        else:
            new_sent.append(tok)
    return new_sent

def printer(sent, should_detokenize=True):
    if should_detokenize:
        sent = detokenize(sent)[1:-1]
    print(" ".join(sent))


# write arbitary string for given sentense. 
import _pickle as cPickle

def grounding(model, question, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask, task_tokens, ):
    vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_prediction, vision_logit, linguisic_prediction, linguisic_logit, attn_data_list, reps = model(
        question, features, spatials, segment_ids, input_mask, image_mask, 
        co_attention_mask, task_tokens, output_all_attention_masks=True, output_representations=True
    )
    logits_vision = torch.max(vision_logit, 1)[1].data
    grounding_val, grounding_idx = torch.sort(vision_logit.view(-1), 0, True)
    return vision_logit, grounding_val, grounding_idx, attn_data_list, reps

def retrieval(model, question, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask, task_tokens, ):
    vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_prediction, vision_logit, linguisic_prediction, linguisic_logit, attn_data_list, reps = model(
        question, features, spatials, segment_ids, input_mask, image_mask, 
        co_attention_mask, task_tokens, output_all_attention_masks=True, output_representations=True
    )
    logits_vil = vil_logit.cpu().numpy()
    logits_vil = np.squeeze(logits_vil)
    return logits_vil, attn_data_list, reps


def prediction(question, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask, task_tokens, ):

    vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_prediction, vision_logit, linguisic_prediction, linguisic_logit, attn_data_list = model(
        question, features, spatials, segment_ids, input_mask, image_mask, 
        co_attention_mask, task_tokens, output_all_attention_masks=True
    )
    
    height, width = img.shape[0], img.shape[1]

    logits = torch.max(vil_prediction, 1)[1].data  # argmax
    # Load VQA label to answers:
    #label2ans_path = os.path.join('save', "VQA" ,"cache", "trainval_label2ans.pkl")
    label2ans_path = os.path.join('cache/trainval_ans2label.pkl')

    vqa_label2ans = cPickle.load(open(label2ans_path, "rb"))
    #answer = vqa_label2ans[logits[0].item()]
    #print("VQA: " + answer)

    # Load GQA label to answers:
    #label2ans_path = os.path.join('save', "gqa" ,"cache", "trainval_label2ans.pkl")
    label2ans_path = os.path.join('cache/trainval_ans2label.pkl')

    logtis_gqa = torch.max(vil_prediction_gqa, 1)[1].data
    gqa_label2ans = cPickle.load(open(label2ans_path, "rb"))

    # vil_binary_prediction NLVR2, 0: False 1: True Task 12
    logtis_binary = torch.max(vil_binary_prediction, 1)[1].data

    # vil_entaliment:  
    label_map = {0:"contradiction", 1:"neutral", 2:"entailment"}
    logtis_tri = torch.max(vil_tri_prediction, 1)[1].data

    # vil_logit: 
    logits_vil = vil_logit[0].item()

    # grounding: 
    logits_vision = torch.max(vision_logit, 1)[1].data
    grounding_val, grounding_idx = torch.sort(vision_logit.view(-1), 0, True)

    examples_per_row = 5
    ncols = examples_per_row 
    nrows = 1
    figsize = [12, ncols*20]     # figure size, inches
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for i, axi in enumerate(ax.flat):
        idx = grounding_idx[i]
        val = grounding_val[i]
        box = spatials[0][idx][:4].tolist()
        y1 = int(box[1] * height)
        y2 = int(box[3] * height)
        x1 = int(box[0] * width)
        x2 = int(box[2] * width)
        patch = img[y1:y2,x1:x2]
        axi.imshow(patch)
        axi.axis('off')
        axi.set_title(str(i) + ": " + str(val.item()))

    plt.axis('off')
    plt.tight_layout(True)
    plt.show() 


def prediction_token(model, question, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask, task_tokens, ):

    sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v, all_attention_mask = model.bert(
        question, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask, task_tokens, output_all_attention_masks=True
    )
    return sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v, all_attention_mask

def custom_prediction(model, query, task, features, infos):

    tokens = tokenizer.encode(query)
    tokens = tokenizer.add_special_tokens_single_sentence(tokens)
    segment_ids = [0] * len(tokens)
    input_mask = [1] * len(tokens)

    max_length = 37
    if len(tokens) < max_length:
        # Note here we pad in front of the sentence
        padding = [0] * (max_length - len(tokens))
        tokens = tokens + padding
        input_mask += padding
        segment_ids += padding

    text = torch.from_numpy(np.array(tokens)).cuda().unsqueeze(0)
    input_mask = torch.from_numpy(np.array(input_mask)).cuda().unsqueeze(0)
    segment_ids = torch.from_numpy(np.array(segment_ids)).cuda().unsqueeze(0)
    task = torch.from_numpy(np.array(task)).cuda().unsqueeze(0)

    num_image = len(infos)

    feature_list = []
    image_location_list = []
    image_mask_list = []
    for i in range(num_image):
        image_w = infos[i]['image_width']
        image_h = infos[i]['image_height']
        feature = features[i]
        num_boxes = feature.shape[0]

        g_feat = torch.sum(feature, dim=0) / num_boxes
        num_boxes = num_boxes + 1
        feature = torch.cat([g_feat.view(1,-1), feature], dim=0)
        boxes = infos[i]['bbox']
        image_location = np.zeros((boxes.shape[0], 5), dtype=np.float32)
        image_location[:,:4] = boxes
        image_location[:,4] = (image_location[:,3] - image_location[:,1]) * (image_location[:,2] - image_location[:,0]) / (float(image_w) * float(image_h))
        image_location[:,0] = image_location[:,0] / float(image_w)
        image_location[:,1] = image_location[:,1] / float(image_h)
        image_location[:,2] = image_location[:,2] / float(image_w)
        image_location[:,3] = image_location[:,3] / float(image_h)
        g_location = np.array([0,0,1,1,1])
        image_location = np.concatenate([np.expand_dims(g_location, axis=0), image_location], axis=0)
        image_mask = [1] * (int(num_boxes))

        feature_list.append(feature)
        image_location_list.append(torch.tensor(image_location))
        image_mask_list.append(torch.tensor(image_mask))

    features = torch.stack(feature_list, dim=0).float().cuda()
    spatials = torch.stack(image_location_list, dim=0).float().cuda()
    image_mask = torch.stack(image_mask_list, dim=0).byte().cuda()
    co_attention_mask = torch.zeros((num_image, num_boxes, max_length)).cuda()

    prediction(text, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask, task)
    return sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v, all_attention_mask

def sent_encode(tokenizer, sent):
    "Encoding as sentence based on the tokenizer"
    sent = sent.strip()
    if sent == "":
        return tokenizer.build_inputs_with_special_tokens([])
    else:
        return tokenizer.encode(sent, add_special_tokens=True)

def bert_score(ref_embedding, hyp_embedding, ref_masks, hyp_masks, use_idf=False):
    batch_size = ref_embedding.size(0)

    ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
    hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))

    sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
    sim = sim[:,1:-1, 1:-1]
    
    masks = torch.bmm(hyp_masks.unsqueeze(2).float(), ref_masks.unsqueeze(1).float())
    masks = masks[:,1:,1:] # default
    masks = masks.expand(batch_size, -1, -1).contiguous().view_as(sim)
    masks = masks.float().to(sim.device)
    sim = sim*masks
    
    word_precision = sim.max(dim=2)[0]
    word_recall = sim.max(dim=1)[0]

    hyp_masks = hyp_masks[:, 1:].float()
    ref_masks = ref_masks[:, 1:].float()

    hyp_masks.div_(hyp_masks.sum(dim=1, keepdim=True))
    ref_masks.div_(ref_masks.sum(dim=1, keepdim=True))
    
    precision_scale = hyp_masks.to(word_precision.device).float()
    recall_scale = ref_masks.to(word_recall.device).float()
    
    P = (word_precision * precision_scale).sum(dim=1)
    R = (word_recall * recall_scale).sum(dim=1)
    F = 2 * P * R / (P + R)
    return P.cpu().numpy(), R.cpu().numpy(), F.cpu().numpy()  