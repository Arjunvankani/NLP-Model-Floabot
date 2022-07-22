#%Run nlp-trainP.py --filename eurotext-eng.txt --epoch 1 --batch 2
from transformers import BertTokenizer, BertForMaskedLM
import torch
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=argparse.FileType('r',encoding="utf-8"),help="File input")
parser.add_argument('--epoch', help="Epoch")
parser.add_argument('--batch', help="Batch size")
args = parser.parse_args()



tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-multilingual-uncased')



lines = args.filename.readline()
#print("lines")

inputs = tokenizer(lines, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
#inputs
#print("input")

inputs['labels'] = inputs.input_ids.detach().clone()
inputs.keys()
#print("input-key")

rand = torch.rand(inputs.input_ids.shape)
mask_arr = (rand < 0.15)*(inputs.input_ids != 101)*(inputs.input_ids != 102) * (inputs.input_ids != 0)
#mask_arr
#print("mask-aa")



selection = []

for i in range(inputs.input_ids.shape[0]):
    selection.append(
        torch.flatten(mask_arr[i].nonzero()).tolist()
    )




for i in range(inputs.input_ids.shape[0]):
    inputs.input_ids[i, selection[i]] = 103

#print("selection")



class MeditationsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
             
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)
    



dataset = MeditationsDataset(inputs)
#print("dataset")
#print(dataset)


loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
device = torch.device('cpu') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.train()


from transformers import AdamW
optim = AdamW(model.parameters(), lr=5e-5)
from tqdm import tqdm  # for our progress bar

#print("epochstart")
for epoch in range(int(args.epoch)):
    loop = tqdm(loader, leave=True)
    for batch in loop:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss
        loss.backward()
        optim.step()
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())


from transformers import TrainingArguments

args = TrainingArguments(
    output_dir='log',
    per_device_train_batch_size=int(args.batch),
    num_train_epochs=int(args.epoch)
)


from transformers import Trainer
#print("trainer")
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset
)


#print("save")
trainer.train()
trainer.save_model('mask-model')    
