import tensorflow as tf
from transformers import AutoTokenizer
import numpy as np
import keras
from function import preprocess_texts
from cnn_model import MedBERTEmbeddingLayer, encode_text, decode_single_text
from similar_model import get_similar_text
import os
import pandas as pd

# Load Data
path = 'training_data.xlsx'
directory_path = os.getcwd()
df = pd.read_excel(os.path.join(directory_path, path))

# tf.keras.utils.register_keras_serializable(MedBERTEmbeddingLayer)


# Load your model with custom layer
with keras.utils.custom_object_scope({'MedBERTEmbeddingLayer': MedBERTEmbeddingLayer}):
    # model = tf.keras.models.load_model('./model.keras')
    model = keras.models.load_model('contoh_model.keras', custom_objects={'MedBERTEmbeddingLayer': MedBERTEmbeddingLayer})



#Tokenizer
tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
input_texts = df['Summary'].tolist()
train_input_ids, train_attention_mask = preprocess_texts(input_texts, tokenizer)


# Model Similarity
vocab_size = 100000  # Example vocabulary size
embedding_dim = 100  # Example embedding dimension
embedding_matrix = tf.random.uniform((vocab_size, embedding_dim))

user_input = str(input('Masukkan Keluhan : '))

input_ids, attention_mask = encode_text(tokenizer, user_input, 128)
predictions = model.predict({'input_ids': input_ids, 'attention_mask': attention_mask})
max_index = np.argmax(predictions)
if (max_index==0):
  category = 'Respiratory siseases'
elif(max_index==1):
  category = 'Skin diseases'
else:
  category = 'Gastrointestinal diseases'
print(category)

df_search = df[df['Category']==category]

real_word = get_similar_text(user_input,tokenizer,train_input_ids,embedding_matrix)

print(real_word)

matching_rows = df_search[df_search['Summary'].str.lower().str.contains(real_word, na=False)]['Drug'].astype(str)

print(matching_rows)