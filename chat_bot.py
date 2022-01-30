from locale import D_FMT
import re
import tensorflow_text
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import pandas as pd
import json
# dataset coronavirus WHO
pd.set_option('max_colwidth', 2000)  # Increase column width
data = pd.read_excel("WHO_FAQ.xlsx", index_col=False)
data.head()
data.reset_index(drop=True, inplace=True)
# Use USE pretrained model to extract response encodings.
# print


def preprocess_sentences(input_sentences):
    return [re.sub(r'(covid-19|covid)', 'coronavirus', input_sentence, flags=re.I)
            for input_sentence in input_sentences]


def predict(message):
    # Load module containing USE
    module = hub.load(
        'https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3')

# Create response embeddings
    response_encodings = module.signatures['response_encoder'](
        input=tf.constant(preprocess_sentences(data.Answer)),
        context=tf.constant(preprocess_sentences(data.Context)))['outputs']

    test_questions = [message]

    # Create encodings for test questions
    question_encodings = module.signatures['question_encoder'](
        tf.constant(preprocess_sentences(test_questions))
    )['outputs']
    # print(question_encodings)
    # Get the responses
    test_responses = data.Answer[np.argmax(
        np.inner(question_encodings, response_encodings), axis=1)]

# Show them in a dataframe
    df = pd.DataFrame({'Test Questions': test_questions,
                       'Test Responses': test_responses})
    # obj = test_responses.to_json()
    # print(obj[1])

    return df["Test Responses"].values.astype(str)[0]


if __name__ == '__main__':
    print(predict("What about pregnant women?"))
