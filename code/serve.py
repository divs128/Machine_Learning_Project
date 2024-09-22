# serve.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertForSequenceClassification, BertTokenizer
import torch

app = FastAPI()

# Load the model and tokenizer from the trained directory
model = BertForSequenceClassification.from_pretrained('./trained_model')
tokenizer = BertTokenizer.from_pretrained('./trained_model')

class ClaimRequest(BaseModel):
    claim_text: str

@app.post("/claim/v1/predict")
async def predict_veracity(claim: ClaimRequest):
    # Tokenize the input claim text
    inputs = tokenizer(claim.claim_text, return_tensors="pt", truncation=True, padding=True)

    # Run the model prediction
    with torch.no_grad():
        logits = model(**inputs).logits

    # Get the predicted class (veracity)
    predicted_class = torch.argmax(logits, dim=1).item()

    # Return the veracity prediction
    return {"veracity": predicted_class}
