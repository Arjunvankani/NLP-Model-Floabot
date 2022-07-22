import torch

from intent_inference import (BertIntentClassifier,
                               AlbertIntentClassifier,
                               DistilBertIntentClassifier,
                               XLMIntentClassifier)
from os.path import join
from typing import (List,
                    Tuple)

class HuggingfaceIntentContainer:
    
    bots = {}
    model_dir = ""
    model_data_dir = ""
    model_data_postfix = ""
    model_postfix = ""

    @classmethod
    def load_bot(cls,
                 bot_id,
                 inference_class,
                 base_model,
                 device='cpu',
                 bot_details=None,
                 model_dir=None,
                 model_data_dir=None,
                 model_data_postfix=None,
                 model_postfix=None,
                 **kargs) -> None:
        
        model_dir = cls.model_dir if not model_dir else model_dir
        model_data_dir = cls.model_data_dir if not model_data_dir else model_data_dir
        model_data_postfix = cls.model_data_postfix if not model_data_postfix else model_data_postfix
        model_postfix = cls.model_postfix if not model_postfix else model_postfix
        model_path = join(model_dir, bot_id + model_postfix)
        csv_path = join(model_data_dir, bot_id + model_data_postfix)
        if bot_id in cls.bots:
            del cls.bots[bot_id]
        cls.bots[bot_id] = inference_class(model_path, csv_path, device)

    @classmethod
    def delete_bot(cls, bot_id) -> None:
        "Deletes bot from bots"
        if bot_id in cls.bots:
            del cls.bots[bot_id]
    unload_bot = delete_bot
    remove_bot = delete_bot

    @classmethod
    def copy_bot(cls,
                 source_id,
                 destination_id,
                 data_path=None,
                 data_postfix=None) -> None:
        
        bot = cls.bots.get(source_id, None)
        if not bot:
            raise RuntimeError(f"{source_id} not loaded yet! Cannot copy it.")
        base_model = bot.base_model
        copied = bot.__class__(base_model=base_model)
        copied.load_bot(bot.model)
        data_path = cls.model_data_dir if not data_path else data_path
        data_postfix = cls.model_data_postfix if not data_postfix else data_postfix
        csv_path = join(data_path, destination_id + data_postfix)
        copied.load_data(csv_path=csv_path)

    @classmethod
    def Response(cls, bot_id, query, preprocess=True) -> List[Tuple[str, float]]:
        
        bot = cls.bots.get(bot_id, None)
        if not bot:
            raise RuntimeError(f"Bot - `{bot_id}` not loaded yet!")

        return bot.infer(query, preprocess)
    
    @classmethod
    def change_class_variables(cls, values: dict) -> None:
        
        model_dir = values.get('model_dir', cls.model_dir)
        model_data_dir = values.get('model_data_dir', cls.model_data_dir)
        model_data_postfix = values.get('model_data_postfix', cls.model_data_postfix)
        model_postfix = values.get('model_postfix', cls.model_postfix)
        cls.model_dir = model_dir
        cls.model_data_dir = model_data_dir
        cls.model_data_postfix = model_data_postfix
        cls.model_postfix = model_postfix

class BertHuggingfaceIntentContainer(HuggingfaceIntentContainer):
    
    bots = {}
    model_dir = "" # Set them when we integrate.
    model_data_dir = ""
    model_data_postfix = ".csv"
    model_postfix = ".pt"

    @classmethod
    def load_bot(cls,
                 bot_id,
                 base_model="bert-base-uncased",
                 device='cpu',
                 model_dir=None,
                 model_data_dir=None,
                 model_data_postfix=None,
                 model_postfix=None,
                 **kargs) -> None:
        
        model_dir = cls.model_dir if not model_dir else model_dir
        model_data_dir = cls.model_data_dir if not model_data_dir else model_data_dir
        model_data_postfix = cls.model_data_postfix if not model_data_postfix else model_data_postfix
        model_postfix = cls.model_postfix if not model_postfix else model_postfix
        model_path = join(model_dir, bot_id + model_postfix)
        csv_path = join(model_data_dir, bot_id + model_data_postfix)
        if bot_id in cls.bots:
            del cls.bots[bot_id]
        cls.bots[bot_id] = BertIntentClassifier(model_path, csv_path, device)
    
    @classmethod
    def Response(cls,
                 bot_id,
                 query,
                 preprocess=True,
                 base_model="bert-base-multilingual-uncased",
                 model_dir=None,
                 model_data_dir=None,
                 model_data_postfix=None,
                 model_postfix=None) -> List[Tuple[str, float]]:
        
        if not bot_id in cls.bots:
            cls.load_bot(bot_id=bot_id,
                         base_model=base_model,
                         model_dir=model_dir,
                         model_data_dir=model_data_dir,
                         model_data_postfix=model_data_postfix,
                         model_postfix=model_postfix)
        bot = cls.bots.get(bot_id)

        return bot.infer(query, preprocess=preprocess)

class AlbertHuggingfaceIntentContainer(HuggingfaceIntentContainer):
    
    bots = {}
    model_dir = "/datadrive/venv1/huggingface/albert_models/" # Set them when we integrate.
    model_data_dir = "/datadrive/venv1/huggingface/"
    model_data_postfix = "_csvfile.csv"
    model_postfix = "_model.pt"

    @classmethod
    def load_bot(cls,
                 bot_id,
                 base_model="albert-base-v1",
                 device='cpu',
                 model_dir=None,
                 model_data_dir=None,
                 model_data_postfix=None,
                 model_postfix=None,
                 **kargs) -> None:
        
        model_dir = cls.model_dir if not model_dir else model_dir
        model_data_dir = cls.model_data_dir if not model_data_dir else model_data_dir
        model_data_postfix = cls.model_data_postfix if not model_data_postfix else model_data_postfix
        model_postfix = cls.model_postfix if not model_postfix else model_postfix
        model_path = join(model_dir, bot_id + model_postfix)
        csv_path = join(model_data_dir, bot_id + model_data_postfix)
        if bot_id in cls.bots:
            del cls.bots[bot_id]
        cls.bots[bot_id] = AlbertIntentClassifier(model_path, csv_path, device)
    
    @classmethod
    def Response(cls,
                 bot_id,
                 query,
                 preprocess=True,
                 base_model="albert-base-v1",
                 model_dir=None,
                 model_data_dir=None,
                 model_data_postfix=None,
                 model_postfix=None) -> List[Tuple[str, float]]:
        
        if not bot_id in cls.bots:
            #print(bot_id, "is not loaded, loading now")
            cls.load_bot(bot_id=bot_id,
                         base_model=base_model,
                         model_dir=model_dir,
                         model_data_dir=model_data_dir,
                         model_data_postfix=model_data_postfix,
                         model_postfix=model_postfix)
        bot = cls.bots.get(bot_id)

        return bot.infer(query, preprocess=preprocess)

class DistilBertHuggingfaceIntentContainer(HuggingfaceIntentContainer):
    
    bots = {}
    model_dir = "" # Set them when we integrate.
    model_data_dir = ""
    model_data_postfix = ""
    model_postfix = ""

    @classmethod
    def load_bot(cls,
                 bot_id,
                 base_model="distilbert-base-uncased",
                 device='cpu',
                 model_dir=None,
                 model_data_dir=None,
                 model_data_postfix=None,
                 model_postfix=None,
                 **kargs) -> None:
        
        model_dir = cls.model_dir if not model_dir else model_dir
        model_data_dir = cls.model_data_dir if not model_data_dir else model_data_dir
        model_data_postfix = cls.model_data_postfix if not model_data_postfix else model_data_postfix
        model_postfix = cls.model_postfix if not model_postfix else model_postfix
        model_path = join(model_dir, bot_id + model_postfix)
        csv_path = join(model_data_dir, bot_id + model_data_postfix)
        if bot_id in cls.bots:
            del cls.bots[bot_id]
        cls.bots[bot_id] = DistilBertIntentClassifier(model_path, csv_path, device)
    
    @classmethod
    def Response(cls,
                 bot_id,
                 query,
                 preprocess=True,
                 base_model="distilbert-base-uncased",
                 model_dir=None,
                 model_data_dir=None,
                 model_data_postfix=None,
                 model_postfix=None) -> List[Tuple[str, float]]:
        
        if not bot_id in cls.bots:
            #print(bot_id, "is not loaded, loading now")
            cls.load_bot(bot_id=bot_id,
                         base_model=base_model,
                         model_dir=model_dir,
                         model_data_dir=model_data_dir,
                         model_data_postfix=model_data_postfix,
                         model_postfix=model_postfix)
        bot = cls.bots.get(bot_id)

        return bot.infer(query, preprocess=preprocess)

class XLMHuggingfaceIntentContainer(HuggingfaceIntentContainer):
    
    bots = {}
    model_dir = "ai/distilbert/trained_models" # Set them when we integrate.
    model_data_dir = "ai/distilbert/training_data"
    model_data_postfix = "_csvfile.csv"
    model_postfix = ".pt"

    @classmethod
    def load_bot(cls,
                 bot_id,
                 base_model="xlm-mlm-100-1280",
                 device='cpu',
                 model_dir=None,
                 model_data_dir=None,
                 model_data_postfix=None,
                 model_postfix=None,
                 **kargs) -> None:
        
        model_dir = cls.model_dir if not model_dir else model_dir
        model_data_dir = cls.model_data_dir if not model_data_dir else model_data_dir
        model_data_postfix = cls.model_data_postfix if not model_data_postfix else model_data_postfix
        model_postfix = cls.model_postfix if not model_postfix else model_postfix
        model_path = join(model_dir, bot_id + model_postfix)
        csv_path = join(model_data_dir, bot_id + model_data_postfix)
        if bot_id in cls.bots:
            del cls.bots[bot_id]
        cls.bots[bot_id] = XLMIntentClassifier(model_path, csv_path, device)
    
    @classmethod
    def Response(cls,
                 bot_id,
                 query,
                 preprocess=True,
                 base_model="xlm-mlm-100-1280",
                 model_dir=None,
                 model_data_dir=None,
                 model_data_postfix=None,
                 model_postfix=None) -> List[Tuple[str, float]]:
        
        if not bot_id in cls.bots:
            #print(bot_id, "is not loaded, loading now")
            cls.load_bot(bot_id=bot_id,
                         base_model=base_model,
                         model_dir=model_dir,
                         model_data_dir=model_data_dir,
                         model_data_postfix=model_data_postfix,
                         model_postfix=model_postfix)
        bot = cls.bots.get(bot_id)

        return bot.infer(query, preprocess=preprocess)

 
#bert = BertHuggingfaceIntentContainer()

#print(bert.Response('final-model',query=" તેમાં પણ તેમણે પોતાના મોક્ષ માટે મહા મહિનાના સુદ પક્ષની આઠમ તિથિને પસંદ કરી હતી."))