# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import jsonlines
import _pickle as cPickle
import logging

import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_transformers.tokenization_bert import BertTokenizer

from ._image_features_reader import ImageFeaturesH5Reader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def _create_entry(item):
    entry = {
        "question_id": item["question_id"],
        "image_id": item["image_id"],
        "hypothesis": item["hypothesis"],
        "scores" : item["scores"]
    }
    return entry


def _load_dataset(dataroot, name, jsonpath, clean_datasets):
    """Load entries

    dataroot: root path of dataset
    name: 'train', 'dev', 'test'
    """
    if name == "train" or name == "dev" or name == "test":
        #annotations_path = os.path.join(dataroot, "qe_%s.jsonl" % name)
        annotations_path = jsonpath
        with jsonlines.open(annotations_path) as reader:
            # Build an index which maps image id with a list of hypothesis annotations.
            items = []
            count = 0
            for annotation in reader:
                # logger.info(annotation)
                dictionary = {}
                dictionary["image_id"] = annotation["image_id"]
                dictionary["question_id"] = count
                dictionary["hypothesis"] = str(annotation["caption"])
                dictionary["scores"] = annotation["score"]
                items.append(dictionary)
                count += 1
    else:
        assert False, "data split is not recognized."

    entries = []
    for item in items:
        entries.append(_create_entry(item))
    return entries


class CaptionQEDataset(Dataset):
    def __init__(
        self,
        task: str,
        dataroot: str,
        annotations_jsonpath: str,
        split: str,
        image_features_reader: ImageFeaturesH5Reader,
        gt_image_features_reader: ImageFeaturesH5Reader,
        tokenizer: BertTokenizer,
        bert_model,
        clean_datasets,
        padding_index: int = 0,
        max_seq_length: int = 16,
        max_region_num: int = 37,
    ):
        super().__init__()
        self.split = split
        self.num_labels = 3
        self._max_region_num = max_region_num
        self._max_seq_length = max_seq_length
        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer
        self._padding_index = padding_index
        self._json_path = annotations_jsonpath
        self.entries = _load_dataset(dataroot, split, self._json_path, clean_datasets)
        self.tokenize(max_seq_length)
        self.tensorize()

    def tokenize(self, max_length=16):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_index in embedding
        """
        for entry in self.entries:
            # tokens = self._tokenizer.tokenize(entry["hypothesis"])
            # tokens = ["[CLS]"] + tokens + ["[SEP]"]

            # tokens = [
            #     self._tokenizer.vocab.get(w, self._tokenizer.vocab["[UNK]"])
            #     for w in tokens
            # ]

            tokens = self._tokenizer.encode(entry["hypothesis"])
            tokens = tokens[: max_length - 2]
            tokens = self._tokenizer.add_special_tokens_single_sentence(tokens)

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (max_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            assert_eq(len(tokens), max_length)
            entry["q_token"] = tokens
            entry["q_input_mask"] = input_mask
            entry["q_segment_ids"] = segment_ids

    def tensorize(self):

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry["q_token"]))
            entry["q_token"] = question

            q_input_mask = torch.from_numpy(np.array(entry["q_input_mask"]))
            entry["q_input_mask"] = q_input_mask

            q_segment_ids = torch.from_numpy(np.array(entry["q_segment_ids"]))
            entry["q_segment_ids"] = q_segment_ids

            if entry["scores"] is not None:
                scores = entry["scores"]
                scores = torch.from_numpy(np.array(scores, dtype=np.float32))
                entry["scores"] = scores
            else:
                entry["scores"] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        image_id = entry["image_id"]
        question_id = entry["question_id"]
        features, num_boxes, boxes, _ = self._image_features_reader[image_id]

        mix_num_boxes = min(int(num_boxes), self._max_region_num)
        mix_boxes_pad = np.zeros((self._max_region_num, 5))
        mix_features_pad = np.zeros((self._max_region_num, 2048))

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)

        mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()

        hypothesis = entry["q_token"]
        input_mask = entry["q_input_mask"]
        segment_ids = entry["q_segment_ids"]

        co_attention_mask = torch.zeros((self._max_region_num, self._max_seq_length))
        #target = torch.zeros(1)
        #scores = answer["scores"]
        target = entry["scores"]

        return (
            features,
            spatials,
            image_mask,
            hypothesis,
            target,
            input_mask,
            segment_ids,
            co_attention_mask,
            question_id,
        )

    def __len__(self):
        return len(self.entries)
