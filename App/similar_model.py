from App.cnn_model import encode_text, decode_single_text
from App.function import standardize_output
import tensorflow as tf

def get_similar_text(user_input, tokenizer, train_input_ids, embedding_matrix):
    encoded_text = encode_text(tokenizer, user_input, 128)
    most_similar_text = get_most_similar_text(encoded_text, train_input_ids, embedding_matrix)
    decoded_text = decode_single_text(most_similar_text, tokenizer)
    real_word = standardize_output(decoded_text)
    return real_word

def get_most_similar_text(text_encoded, train_input_ids, embedding_matrix):
    max_jaccard_similarity = -1  # Initialize with lowest possible similarity
    most_similar_text = None

    # Convert encoded text to embeddings (ensure correct dtype)
    embedding_text_encoded = tf.nn.embedding_lookup(embedding_matrix, text_encoded)
    embedding_text_encoded = tf.cast(embedding_text_encoded, dtype=tf.float32)

    for i in train_input_ids:
        # Convert tokenized sequence to embeddings (ensure correct dtype)
        embedding_text_train = tf.nn.embedding_lookup(embedding_matrix, i)
        embedding_text_train = tf.cast(embedding_text_train, dtype=tf.float32)

        # Compute Jaccard similarity with embedded representations
        jaccard_sim = jaccard_similarity(embedding_text_encoded, embedding_text_train)

        # Update most similar text if Jaccard similarity is higher
        if jaccard_sim > max_jaccard_similarity:
            max_jaccard_similarity = jaccard_sim
            most_similar_text = i

    return most_similar_text

def jaccard_similarity(embedded_doc1, embedded_doc2):
    # Convert tensors to sets of tuples

    embedded_doc1 = tf.constant(embedded_doc1)
    embedded_doc2 = tf.constant(embedded_doc2)

    # Convert the tensors to numpy arrays
    numpy_doc1 = embedded_doc1.numpy()
    numpy_doc2 = embedded_doc2.numpy()

    # Flatten the numpy arrays to 2D if needed
    flattened_doc1 = numpy_doc1.reshape(-1, numpy_doc1.shape[-1])
    flattened_doc2 = numpy_doc2.reshape(-1, numpy_doc2.shape[-1])

    # Convert numpy arrays to sets of tuples
    set1 = set(tuple(row) for row in flattened_doc1)
    set2 = set(tuple(row) for row in flattened_doc2)
    # Calculate Jaccard similarity
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    jaccard_sim = intersection / union

    return jaccard_sim