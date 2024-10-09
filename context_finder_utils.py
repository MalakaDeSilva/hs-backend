from transformers import BertTokenizer
import tensorflow_hub as hub

import tensorflow as tf

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

labse_model = hub.KerasLayer("https://www.kaggle.com/models/google/labse/TensorFlow2/labse/2")

classes = ['age', 'body shaming', 'class', 'disability', 'gender', 'none',
           'political', 'racial', 'religion', 'sexual', 'threat']


# Tokenize the texts
def tokenize_texts(texts, max_len=128):
    return tokenizer(
        list(texts),
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )


# Get embeddings from LaBSE
def embed_texts_for_ctx_extractor(tokens):
    embeddings = labse_model({
        "input_word_ids": tokens['input_ids'],
        "input_mask": tokens['attention_mask'],
        "input_type_ids": tokens['token_type_ids']
    })
    return embeddings['default']


def extract_context(predictions):
    # Get the predicted class index
    predicted_class_indices = tf.argmax(predictions, axis=1).numpy()
    predicted_labels = classes[predicted_class_indices[0]]
    return predicted_labels[0]
