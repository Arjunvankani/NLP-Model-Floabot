from transformers import BertTokenizer, BertForMaskedLM
import torch
import argparse
import os
import time
import sys
import pickle
from transformers import AdamW
from tqdm import tqdm 
from transformers import TrainingArguments
from transformers import Trainer


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-multilingual-uncased')
class MeditationsDataset(torch.utils.data.Dataset):
    def __init__(self, lines):
       
        self.lines = lines
    def __getitem__(self, idx):

        inputs = tokenizer(self.lines[idx], return_tensors='pt', max_length=64, truncation=True, padding='max_length')
        inputs['labels'] = inputs.input_ids.detach()
        rand = torch.rand(inputs.input_ids.shape)
        mask_arr = (rand < 0.15)*(inputs.input_ids != 101)*(inputs.input_ids != 102)*(inputs.input_ids != 0)
        selection = []
        for i in range(inputs.input_ids.shape[0]):
            selection.append(torch.flatten(mask_arr[i].nonzero()).tolist())
        for i in range(inputs.input_ids.shape[0]):
            inputs.input_ids[i, selection[i]] = 103
        return {key: torch.squeeze(torch.tensor(val)) for key, val in inputs.items()}
    def __len__(self):
        return len(self.lines)
    

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=argparse.FileType('r',encoding="utf-8"),help="File input")
    parser.add_argument('--epoch', help="Epoch", default=2)
    parser.add_argument('--batch', help="Batch size",default=4)
    parser.add_argument('--output', help="output location")
    args1 = parser.parse_args()

    lines = args1.filename.readlines()
    dataset = MeditationsDataset(lines)
    loader = torch.utils.data.DataLoader(dataset, batch_size=int(args1.batch), shuffle=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()



    optim = AdamW(model.parameters(), lr=5e-5)
    args = TrainingArguments(
        output_dir='out',
        per_device_train_batch_size=int(args1.batch),
        num_train_epochs=int(args1.epoch),
        logging_steps=int(args1.epoch)/2,            
        save_steps=int(args1.epoch)/2
    )


    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
    )


    trainer.train()
    trainer.save_model(args1.output+'/mask-model')    


if __name__ == "__main__":
    main()
    
