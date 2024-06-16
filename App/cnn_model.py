from transformers import AutoTokenizer, TFAutoModel
from tensorflow import keras
from keras import Layer
from keras.regularizers import l2
import tensorflow as tf

class MedBERTEmbeddingLayer(Layer):
    def __init__(self, model_name='emilyalsentzer/Bio_ClinicalBERT', **kwargs):
        super(MedBERTEmbeddingLayer, self).__init__(**kwargs)
        # self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = TFAutoModel.from_pretrained(model_name)

    def call(self, inputs):
        input_ids, attention_mask = inputs
        outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

    # def get_config(self):
    #     config = super(MedBERTEmbeddingLayer, self).get_config()
    #     config.update({
    #         'model_name': self.model_name,
    #     })
    #     return config

    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)
    
def create_text_classification_model(max_length=128):
    input_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name='attention_mask')

    medbert_layer = MedBERTEmbeddingLayer()
    token_embeddings = medbert_layer((input_ids, attention_mask))

    conv5 = tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu', kernel_regularizer=l2(0.01))(token_embeddings)
    pool5 = tf.keras.layers.MaxPooling1D(pool_size=3)(conv5)
    conv3 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01))(pool5)
    pool3 = tf.keras.layers.GlobalMaxPooling1D()(conv3)

    dense1 = tf.keras.layers.Dense(64, activation='relu')(pool3)
    dropout1 = tf.keras.layers.Dropout(0.2)(dense1)
    dense2 = tf.keras.layers.Dense(32, activation='relu')(dropout1)
    dropout2 = tf.keras.layers.Dropout(0.2)(dense2)
    output = tf.keras.layers.Dense(3, activation='softmax')(dropout2)

    model = tf.keras.models.Model(inputs=[input_ids, attention_mask], outputs=output)
    return model

def encode_text(tokenizer, text, max_length):
    encoding = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='tf'
    )
    return encoding['input_ids'],encoding['attention_mask']

def decode_single_text(input_ids, tokenizer):
    decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
    return decoded_text

class Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs.get("accuracy")
        val_accuracy = logs.get("val_accuracy")
        if (accuracy >= 0.81 and val_accuracy >= 0.81):
            print(f"\nStopping training have reached above 81% ({accuracy:.4f}) and validation accuracy ({val_accuracy:.4f}) have reached above 81%")
            self.model.stop_training = True
