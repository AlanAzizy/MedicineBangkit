import os
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from keras.models import load_model
from App.function import preprocess_texts
from App.cnn_model import MedBERTEmbeddingLayer, encode_text
from App.similar_model import get_similar_text

app = FastAPI()

# Load Data
path = './App/training_data.xlsx'
directory_path = os.getcwd()
df = pd.read_excel(os.path.join(directory_path, path))

model_path = './model/model.keras'
print(os.path.join(directory_path, model_path))

# Load your model with custom layer
with tf.keras.utils.custom_object_scope({'MedBERTEmbeddingLayer': MedBERTEmbeddingLayer}):
    model = load_model(os.path.join(directory_path, model_path), custom_objects={'MedBERTEmbeddingLayer': MedBERTEmbeddingLayer})

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
input_texts = df['Summary'].tolist()
train_input_ids, train_attention_mask = preprocess_texts(input_texts, tokenizer)

# Model Similarity Parameters
vocab_size = 100000  # Example vocabulary size
embedding_dim = 100  # Example embedding dimension
embedding_matrix = tf.random.uniform((vocab_size, embedding_dim))

class UserInput(BaseModel):
    keluhan: str

@app.post("/predict")
async def predict(user_input: UserInput):
    input_text = user_input.keluhan

    # Encode user input
    input_ids, attention_mask = encode_text(tokenizer, input_text, 128)
    predictions = model.predict({'input_ids': input_ids, 'attention_mask': attention_mask})
    max_index = np.argmax(predictions)
    
    if max_index == 0:
        category = 'Gastrointestinal diseases'
    elif max_index == 1:
        category = 'Respiratory diseases'
    else:
        category = 'Skin diseases'
    
    df_search = df[df['Category'] == category]
    
    # Find similar text
    real_word = get_similar_text(input_text, tokenizer, train_input_ids, embedding_matrix)
    
    # Find matching drugs
    matching_rows = df_search[df_search['Summary'].str.lower().str.contains(real_word, na=False)]['Drug'].astype(str).tolist()
    
    if not matching_rows:
        raise HTTPException(status_code=404, detail="No matching drugs found")
    
    return {"category": category, "drugs": matching_rows}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
