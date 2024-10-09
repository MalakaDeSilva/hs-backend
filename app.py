import joblib
from flask import Flask, jsonify, request
from tensorflow.python.keras.models import model_from_json

from context_finder_utils import tokenize_texts, embed_texts_for_ctx_extractor, extract_context
from embedding_utils import embed_texts_for_detector

app = Flask(__name__)

# Load the saved model architecture and weights
with open('hate_speech_detector.pkl', 'rb') as f:
    classifier_dict = joblib.load(f)

# Recreate the model architecture from the JSON string
classifier = model_from_json(classifier_dict['architecture'])

# Load the weights into the model
classifier.set_weights(classifier_dict['weights'])

# Load the saved model architecture and weights
with open('context_finder.pkl', 'rb') as f:
    context_finder_dict = joblib.load(f)

# Recreate the model architecture from the JSON string
context_finder = model_from_json(context_finder_dict['architecture'])

# Load the weights into the model
context_finder.set_weights(context_finder_dict['weights'])

tweets = []


# Route to get all tweets
@app.route('/tweets', methods=['GET'])
def get_tweets():
    return jsonify(tweets)


# Route to get a specific tweet by ID
@app.route('/tweets/<int:tweet_id>', methods=['GET'])
def get_tweet(tweet_id):
    tweet = next((tweet for tweet in tweets if tweet['id'] == tweet_id), None)
    if tweet:
        return jsonify(tweet)
    return jsonify({'message': 'Tweet not found'}), 404


# Route to create a new tweet
@app.route('/tweets', methods=['POST'])
def create_tweet():
    new_tweet = request.get_json()
    new_tweet['id'] = len(tweets) + 1

    # Classify hate speech / non-hate speech
    new_texts = [new_tweet.content]
    new_texts_embeddings = embed_texts_for_detector(new_texts)
    predictions = classifier.predict(new_texts_embeddings)

    # Find context
    if predictions[0] > 0.5:
        new_tokens = tokenize_texts(new_texts)
        new_embeddings = embed_texts_for_ctx_extractor(new_tokens)
        prediction = context_finder.predict(new_embeddings)
        context = extract_context(prediction)

        new_tweet["detected"] = True
        new_tweet["context"] = context[0]
    else:
        new_tweet["detected"] = False

    return jsonify(new_tweet), 201


if __name__ == '__main__':
    app.run(debug=True)
