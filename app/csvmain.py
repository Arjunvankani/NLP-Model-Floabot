import uvicorn
from fastapi import File, UploadFile, FastAPI
from typing import List
import csv
import intent_inference_containers
import codecs
import pandas as pd
app = FastAPI()

bert = intent_inference_containers.BertHuggingfaceIntentContainer()


def save_file(filename, data):
    with open(filename, 'wb') as f:
        f.write(data)

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    
    # in case you need the files saved, once they are uploaded
    for file in files:
       
        data = file.file
        data = csv.reader(codecs.iterdecode(data,'utf-8-sig'), delimiter='\n')
        for i in data:
            df = pd.DataFrame(data)
            for j in df[0]:

                result = bert.Response('final-model',query=j)
                print(j,result)
               
                
        
        
        print('--')


    return {"result" : df}
    

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000, debug=True)
    


    
    
    
    