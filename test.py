import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

# Ensure TensorFlow 2.x behavior
print("TensorFlow version:", tf.__version__)
print("TensorFlow Hub version:", hub.__version__)

# Load preprocessor and encoder
preprocessor = hub.KerasLayer(hub.load("./preprocessor"))
encoder = hub.KerasLayer(hub.load("./labse"))

def embed_texts(texts):
    preprocessed_texts = preprocessor(texts)
    return encoder(preprocessed_texts)["default"]


# Example usage
texts = ["Hello world!"]
embeddings = embed_texts(texts)
print(embeddings)
