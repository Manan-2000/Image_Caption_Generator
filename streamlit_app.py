import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.models import Model, load_model # type: ignore
import pickle

# Load the preprocessed mapping and tokenizer
with open("/workspaces/Image_Caption_Generator/mapping.pkl", "rb") as f:
    mapping = pickle.load(f)

def all_captions(mapping):
    return [caption for key in mapping for caption in mapping[key]]
captions = all_captions(mapping)

def create_token(all_captions):
    from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    return tokenizer

tokenizer = create_token(captions)
max_length = 35

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    repeated_word_count = 0
    previous_word = None
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)

        # Break if we hit the end token or repeat the same word too much
        if word is None or word == 'endseq' or (word == previous_word and repeated_word_count > 2):
            break

        in_text += " " + word

        # Track the repeated word
        if word == previous_word:
            repeated_word_count += 1
        else:
            repeated_word_count = 0
        previous_word = word

    return in_text.strip('startseq ').strip()

# Load the VGG16 model and the captioning model
vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
model = load_model("/workspaces/Image_Caption_Generator/model.keras")

# Streamlit app
st.title("Image Caption Generator")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = load_img(uploaded_file, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = vgg_model.predict(image, verbose=0)

    caption = predict_caption(model, feature, tokenizer, max_length)
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Generated Caption: ", caption)
