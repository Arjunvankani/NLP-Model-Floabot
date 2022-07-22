#uvicorn main:app --reload
from fastapi import FastAPI
import intent_inference_containers 
app = FastAPI()


bert = intent_inference_containers.BertHuggingfaceIntentContainer()

@app.get("/items/{bot_id}")

async def read_item(bot_id):
    
    result = bert.Response('final-model',query=bot_id)
    print(result)
    return {"bot_id": bot_id,"result" : result}


#print(bert.Response('final-model',query=" મોદી સરકાર 1 ફેબ્રુઆરીએ વચગાળાનું બજેટ રજૂ કરશે, આ જાહેરાતો થઈ શકે"))
