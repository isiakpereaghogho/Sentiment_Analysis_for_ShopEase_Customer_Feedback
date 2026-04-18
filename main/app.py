import io
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
from pipelines.prediction import PredictSentiment
from pipelines.training import train_and_evaluate

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = FastAPI(title="ShopEase Sentiment API")

# Pydantic schema for single-text input
class TextRequest(BaseModel):
    text: str

predictor = PredictSentiment()
logging.info("Model loaded successfully.")


@app.post("/predict_sentiment")
def predict_text(request: TextRequest):
    try:
        result = predictor.predict(request.text)

        return {
            "sentiment": result["sentiment"],
            "confidence": float(result["confidence"]),
            "scores": result["scores"]
        }

    except Exception as e:
        logging.error(f"Error occurred while predicting the sentiment: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        if "review" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain a 'review' column.")

        result_list = []

        for idx, row in df.iterrows():
            try:
                review = str(row["review"])
                result = predictor.predict(review)

                result_row = row.to_dict()
                result_row["sentiment"] = result["sentiment"]
                result_row["confidence"] = float(result["confidence"])

                # optional: include all scores
                result_row["negative_score"] = float(result["scores"].get("negative", 0))
                result_row["neutral_score"] = float(result["scores"].get("neutral", 0))
                result_row["positive_score"] = float(result["scores"].get("positive", 0))

                result_list.append(result_row)

            except Exception as e:
                logging.error(f"Error occurred while predicting sentiment for review {idx}: {e}")
                continue

        return result_list

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error occurred during batch prediction: {e}")
        raise HTTPException(status_code=500, detail="Batch prediction failed")
    
@app.get("/train")
def train_model():
    try:
        train_and_evaluate()
        return {"message": "Training completed successfully"}
    except Exception as e:
        logging.error(f"Error occurred during training: {e}")
        raise HTTPException(status_code=500, detail="Training failed")