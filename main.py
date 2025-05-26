from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# carregar modelo e vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# instancia da api
app = FastAPI()

# schema
class Review(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(review: Review):
    vect_text = vectorizer.transform([review.text])
    prediction = model.predict(vect_text)
    sentiment = "Positive" if prediction[0] == 1 else "Negative"
    return {"sentiment": sentiment}
