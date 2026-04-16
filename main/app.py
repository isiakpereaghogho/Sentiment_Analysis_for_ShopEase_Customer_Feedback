import io
import logging
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel  
import pandas as pd
import numpy as np
from pipelines.prediction import predict_sentiment
from pipelines.training import train_and_evaluate   

logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(message)s'
    )

app = FastAPI(title="ShopEase Sentiment API")

#pydantic schema for input dataset
class TextRequest(BaseModel):
    text: str

predictor = predict_sentiment()
logging.info("Model loaded successfully.")

app.post("/predict_sentiment")
def predict_text(request: TextRequest):
    try:
        result = predictor.predict(request.text)
        print(result)
        top_label = max(result, key=lambda x: x['score'])
        return {"label": top_label['label'], "confidencescore": float(top_label['score'])}
    except Exception as e:
        logging.error(f"Error occurred while predicting the sentiment: {e}")

app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    contents = await file.read()
    
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    if "reviews" not in df.columns:
        return {"error": "CSV must contain a 'review' column."}
    
    #predict sentiment for each review
    result_list = []
    for idx, row in df.iterrows():
        try:
            review = str(row['review'])
            result = predictor.predict(review)

            if result is None or len(result) == 0:
                raise ValueError(f"No prediction returned for review: {review}") #Empty result from the model
            
            top_label = max(result, key=lambda x: x['score'])
            result_row = row.to_dict()
            result_row["sentiment_label"] = top_label['label']
            result_row["confidence_score"] = float(top_label['score'])
            result_list.append(result_row)

        except Exception as e:
            logging.error(f"Error occurred while predicting sentiment for review {idx}: {e}")
            continue
        return {"predictions": result_list} 