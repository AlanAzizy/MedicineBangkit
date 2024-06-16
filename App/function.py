import pandas as pd
import matplotlib.pyplot as plt

def standardize_output(text):
  filter_1 = text.replace('(', '')
  filter_2 = filter_1.replace(')','')
  filter_3 = filter_2.replace('-', '')
  filter_4 = filter_3.replace('   ','')
  return filter_4.replace('  ',' ')

def split_row_by_column(df, column, delimiter):
    new_rows = []

    for index, row in df.iterrows():
        # Split the column value based on the delimiter
        if (type(row[column])==str):
          split_values = row[column].split(delimiter)

          # Create new rows from the split values
          for value in split_values:
              new_row = row.copy()
              new_row[column] = value.strip()  # Remove leading/trailing whitespace
              new_rows.append(new_row)

    # Create a new DataFrame from the new rows
    new_df = pd.DataFrame(new_rows)
    return new_df

def preprocess_texts(texts, tokenizer, max_length=128):
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    encoding = tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='tf')
    return encoding['input_ids'], encoding['attention_mask']

def plot_training_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()
