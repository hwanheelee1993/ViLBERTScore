# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .concept_cap_dataset import (
    ConceptCapLoaderTrain,
    ConceptCapLoaderVal,
    ConceptCapLoaderRetrieval,
)
from .foil_dataset import FoilClassificationDataset
from .vqa_dataset import VQAClassificationDataset
from .vqa_mc_dataset import VQAMultipleChoiceDataset
from .nlvr2_dataset import NLVR2Dataset
from .retreival_dataset import RetreivalDataset, RetreivalDatasetVal
from .vcr_dataset import VCRDataset
from .visdial_dataset import VisDialDataset
from .visual_entailment_dataset import VisualEntailmentDataset
from .refer_dense_caption import ReferDenseCpationDataset
from .visual_genome_dataset import GenomeQAClassificationDataset
from .gqa_dataset import GQAClassificationDataset
from .guesswhat_dataset import GuessWhatDataset
from .visual7w_pointing_dataset import Visual7wPointingDataset
from .guesswhat_pointing_dataset import GuessWhatPointingDataset
from .flickr_grounding_dataset import FlickrGroundingDataset
from .captionqe_dataset import CaptionQEDataset

# from .flickr_retreival_dataset import FlickrRetreivalDatasetTrain, FlickrRetreivalDatasetVal
__all__ = [
    "FoilClassificationDataset",
    "VQAClassificationDataset",
    "GenomeQAClassificationDataset",
    "VQAMultipleChoiceDataset",
    "ConceptCapLoaderTrain",
    "ConceptCapLoaderVal",
    "NLVR2Dataset",
    "ReferExpressionDataset",
    "RetreivalDataset",
    "RetreivalDatasetVal",
    "VCRDataset",
    "VisDialDataset",
    "VisualEntailmentDataset",
    "GQAClassificationDataset",
    "ConceptCapLoaderRetrieval",
    "GuessWhatDataset",
    "Visual7wPointingDataset",
    "GuessWhatPointingDataset",
    "FlickrGroundingDataset",
    "CaptionQEDataset",
    "",
]

DatasetMapTrain = {
    "VQA": VQAClassificationDataset,
    "GenomeQA": GenomeQAClassificationDataset,
    "VisualDialog": VisDialDataset,
    "VCR_Q-A": VCRDataset,
    "VCR_QA-R": VCRDataset,
    "RetrievalCOCO": RetreivalDataset,
    "RetrievalFlickr30k": RetreivalDataset,
    "NLVR2": NLVR2Dataset,
    "VisualEntailment": VisualEntailmentDataset,
    "GQA": GQAClassificationDataset,
    "Foil": FoilClassificationDataset,
    "GuessWhat": GuessWhatDataset,
    "Visual7w": Visual7wPointingDataset,
    "GuessWhatPointing": GuessWhatPointingDataset,
    "FlickrGrounding": FlickrGroundingDataset,
    "CaptionQE" : CaptionQEDataset
}


DatasetMapEval = {
    "VQA": VQAClassificationDataset,
    "GenomeQA": GenomeQAClassificationDataset,
    "VisualDialog": VisDialDataset,
    "VCR_Q-A": VCRDataset,
    "VCR_QA-R": VCRDataset,
    "RetrievalCOCO": RetreivalDatasetVal,
    "RetrievalFlickr30k": RetreivalDatasetVal,
    "NLVR2": NLVR2Dataset,
    "VisualEntailment": VisualEntailmentDataset,
    "GQA": GQAClassificationDataset,
    "Foil": FoilClassificationDataset,
    "GuessWhat": GuessWhatDataset,
    "Visual7w": Visual7wPointingDataset,
    "GuessWhatPointing": GuessWhatPointingDataset,
    "FlickrGrounding": FlickrGroundingDataset,
    "CaptionQE" : CaptionQEDataset
}
