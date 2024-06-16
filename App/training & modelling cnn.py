import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
from keras.callbacks import ReduceLROnPlateau
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
import numpy as np
from cnn_model import create_text_classification_model, Callback
from function import preprocess_texts, plot_training_history


# Load Data
import pandas as pd
df = pd.read_excel('./training_data.xlsx',0)

#Prepare Data
df = df.groupby('Category').apply(lambda x: x.sample(n=970, random_state=4)).reset_index(drop=True)

#Labelling
floatLabel = df.pop('Category')

if __name__ == "__main__":
    model = create_text_classification_model()
    opt = keras.optimizers.Adam(learning_rate=0.00005)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    input_texts = df['Summary'].tolist()

    #  Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(floatLabel)

    # Split the dataset into training and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(input_texts, encoded_labels, test_size=0.3, random_state=42)
    print(train_texts[0])
    # Tokenize input texts
    train_input_ids, train_attention_mask = preprocess_texts(train_texts, tokenizer)
    val_input_ids, val_attention_mask = preprocess_texts(val_texts, tokenizer)
    print(train_input_ids[0])
    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': train_input_ids, 'attention_mask': train_attention_mask}, train_labels)).batch(20)
    val_dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': val_input_ids, 'attention_mask': val_attention_mask}, val_labels)).batch(20)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00005)

    callback = Callback()

    # Fit the model with validation data
    history = model.fit(train_dataset, epochs=100, validation_data=val_dataset, callbacks=[reduce_lr, callback])

    plot_training_history(history)

    model.save('model.h5')
    model.save('model.keras')