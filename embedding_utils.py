import tensorflow_hub as hub
import tensorflow_text as text

# Load the preprocessor
# Load preprocessor and encoder
encoder = hub.KerasLayer("https://www.kaggle.com/models/google/labse/TensorFlow2/labse/2")
preprocessor = hub.KerasLayer("https://kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/cmlm-multilingual-preprocess/2")


def embed_texts_for_detector(texts):
    """
    Convert a list of texts to their embeddings.

    Args:
    texts (list of str): List of strings to be embedded.

    Returns:
    numpy.ndarray: Array of embeddings corresponding to the input texts.
    """
    preprocessed_texts = preprocessor(texts)
    return encoder(preprocessed_texts)["default"].numpy()
