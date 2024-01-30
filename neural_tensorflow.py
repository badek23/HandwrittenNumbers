
#### Import Libraries ####
import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf


#### Streamlit ####
st.set_page_config(page_title='Neural Network', page_icon='🧠')

# Hide menu + footer options for users
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

# Text on page
st.markdown("<h1 style='text-align: center; color: black;'>Handwritten Number Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='font-size: 1em; text-align: center; color: black;'>Write a number below on the canvas, and our algorithm will predict which number it is.</h1>", unsafe_allow_html=True)

canvas_result = st_canvas(
    stroke_width=10,
    stroke_color='#ffffff',
    background_color="#00000",
    height=150,
    width=150,
    drawing_mode='freedraw',
    key="canvas",
)

#### Prediction ####

# Load model
model = tf.keras.models.load_model("trained_params_tensorflow")

if canvas_result.image_data is not None:
    st.image(canvas_result.image_data)

    # Make prediction on single image
    image = Image.fromarray(canvas_result.image_data).convert('L')
    image = image.resize((28, 28))
    image = np.array(image)

    # Put into collection where it's the only member
    image = (np.expand_dims(image,0))

    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    # Predict
    prediction = probability_model.predict(image)

    # Choose most likely prediction
    prediction_max = np.argmax(prediction[0])

    if image.any() != 0:
        st.write('Your image is:', prediction_max)

