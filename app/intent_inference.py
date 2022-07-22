import torch
import transformers
import re
import string

import pandas as pd

from torch.nn.functional import softmax
from transformers import (BertTokenizer,
                          AlbertTokenizer,
                          DistilBertTokenizer,
                          XLMTokenizer)
from sklearn.preprocessing import LabelEncoder
import numpy as np

class DummyEncoder:
    
    def __init__(self, data):
        ls = []
        ids = []
        for row in data[[1, 2]].drop_duplicates().sort_values(by=2).iterrows():
            ls.append(row[1][1])
            ids.append(row[1][2])
        ls = np.array(ls)
        ids = np.array(ids)
        self.data = data
        self.classes_ = ls
        self.ids_ = ids
    
    def inverse_transform(self, index_in_list):
        return [self.classes_[index_in_list[0]]]

def wordExpand(text):
    contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "I had ",
    "i'd've": "I would have",
    "i'll": "I will",
    "i'll've": "I will have",
    "i'm": "I am",
    "i've": "I have",
    "isn't": "is not",
    "it'd": "it had",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that has",
    "there'd": "there had",
    "there'd've": "there would have",
    "there's": "there has",
    "they'd": "they had",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when has",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who shall",
    "who'll've": "who shall have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why has",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had",
    "you'd've": "you would have",
    "you'll": "you shall",
    "you'll've": "you shall have",
    "you're": "you are",
    "you've": "you have"
    }
    
    for word in text.split():
        if word.lower() in contractions:
            text = text.replace(word, contractions[word.lower()])
    return text

def preprocessQuery(query):
    
    query = query.strip()
    query = wordExpand(query)
    query = re.split("\W+", query)
    query = " ".join([word.lower() for word in query if word not in string.punctuation])
    query = query.strip()
    return query

from pathlib import Path
from typing import Union

class IntentClassifier:
    
    def __init__(self,
                 model: Union[str, Path, None]=None,
                 csv: Union[str, Path, None]=None,
                 device: Union[str, None]=None):
        self._change_device(device)
        if not model:
            self.model = None
        else:
            self.load_bot(model)
        if not csv:
            self.le = None
        else:
            self.load_data(csv)
        self.base_model = None
        self.tokenizer = None

    def _change_device(self,
                      device: str='cpu'):
        if not device:
            self.device = torch.device('cpu')
            return
        if not (device.startswith('cpu') or device.startswith('cuda')):
            raise ValueError("You can load the model only on `cpu` or `cuda`. Invalid Argument:", device)
        self.device = torch.device(device)
        # if self.model:
        #     self.model = self.model.to(device)
    
    def change_device(self,
                      device: str='cpu'):
        if not (device.startswith('cpu') or device.startswith('cuda')):
            raise ValueError("You can load the model only on `cpu` or `cuda`. Invalid Argument:", device)
        self.device = torch.device(device)
        if self.model:
            self.model = self.model.to(device)
    
    def load_bot(self,
                 model: Union[str, Path]):
        
        if type(model) in (str, Path):
            self.model = torch.load(model, map_location=self.device)
            self.model = self.model.eval()
            return
        self.model = model
    
    def load_data(self,
                  csv_path: Union[str, Path]):
        
        data = pd.read_csv(csv_path, sep=',', header=None)
        if 2 not in data.columns:
            le = LabelEncoder()
            data[2] = le.fit_transform(data[1])
        else:
            le = DummyEncoder(data)
        self.le = le
    
    def _infer(self, pred_input):
        
        return self.model(**pred_input)
    
    def infer(self, query: str, preprocess=True):
        
        if not self.le or not self.model or not self.tokenizer:
            raise RuntimeError("You need to load the model first.")
        if preprocess:
            query = preprocessQuery(query)
        
        pred_input = self.tokenizer(query,
                                    return_tensors="pt",
                                    padding=True,
                                    truncation=True)
        
        pred_output = self._infer(pred_input=pred_input)
        np_tensor = pred_output[0][0].detach().numpy()
        index = np_tensor.argmax()
        pred = self.le.inverse_transform([index])
        prob = softmax(pred_output[0], dim=1).max().item()

        return [(pred[0], prob)]

class BertIntentClassifier(IntentClassifier):
    
    def __init__(self,
                 model: Union[str, Path, None]=None,
                 csv: Union[str, Path, None]=None,
                 device: Union[str, None]=None,
                 base_model: str="bert-base-uncased"):
        IntentClassifier.__init__(self, model, csv, device)
        self.base_model = base_model
        self.tokenizer = BertTokenizer.from_pretrained(base_model)

class AlbertIntentClassifier(IntentClassifier):
    
    def __init__(self,
                 model: Union[str, Path, None]=None,
                 csv: Union[str, Path, None]=None,
                 device: Union[str, None]=None,
                 base_model: str="albert-base-v1"):
        IntentClassifier.__init__(self, model, csv, device)
        self.base_model = base_model
        self.tokenizer = AlbertTokenizer.from_pretrained(base_model)

class DistilBertIntentClassifier(IntentClassifier):
    
    def __init__(self,
                 model: Union[str, Path, None]=None,
                 csv: Union[str, Path, None]=None,
                 device: Union[str, None]=None,
                 base_model: str="distilbert-base-uncased"):
        IntentClassifier.__init__(self, model, csv, device)
        self.base_model = base_model
        self.tokenizer = DistilBertTokenizer.from_pretrained(base_model)
    
class XLMIntentClassifier(IntentClassifier):
    def __init__(self,
                 model: Union[str, Path, None]=None,
                 csv: Union[str, Path, None]=None,
                 device: Union[str, None]=None,
                 base_model: str="xlm-mlm-100-1280"):
        IntentClassifier.__init__(self, model, csv, device)
        self.base_model = base_model
        self.tokenizer = XLMTokenizer.from_pretrained(base_model)