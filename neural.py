
#### Import Libraries ####
import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import pickle
import matplotlib.pyplot as plt


#### Neural Network Functions ####
def ReLU(Z):
    return np.maximum(Z,0)

def softmax(Z):
    """Compute softmax values for each sets of scores in x."""
    exp = np.exp(Z - np.max(Z))
    return exp / exp.sum(axis=0)

def init_params(size):
    W1 = np.random.normal(size=(10, 784)) * np.sqrt(1./(784))
    b1 = np.random.normal(size=(10, 1)) * np.sqrt(1./10)
    W2 = np.random.normal(size=(10, 10)) * np.sqrt(1./20)
    b2 = np.random.normal(size=(10, 1)) * np.sqrt(1./(784))
    return W1,b1,W2,b2

def forward_propagation(X,W1,b1,W2,b2):
    Z1 = W1.dot(X) + b1 #10, m
    A1 = ReLU(Z1) # 10,m
    Z2 = W2.dot(A1) + b2 #10,m
    A2 = softmax(Z2) #10,m
    return Z1, A1, Z2, A2

def get_predictions(A2):
    return np.argmax(A2, 0)

def make_predictions(image, W1, b1, W2, b2):
    # Flatten the image and normalize
    vect_X = image.flatten() / 255
    vect_X = vect_X[:, None]

    Z1, A1, Z2, A2 = forward_propagation(vect_X, W1, b1, W2, b2)
    prediction = get_predictions(A2)
    return prediction

def show_prediction(image, W1, b1, W2, b2):
    prediction = make_predictions(image, W1, b1, W2, b2)
    print("Prediction: ", prediction)
    plt.gray()
    #plt.imshow(current_image, interpolation='nearest')
    plt.show()
    return prediction


# Open trained params
with open("trained_params.pkl","rb") as dump_file:
    W1, b1, W2, b2=pickle.load(dump_file)


#### Streamlit ####
st.set_page_config(page_title='Neural Network', page_icon='🧠')

# Hide menu + footer options for users
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

# Text on page
st.title('Handwritten Number Predictor')
st.markdown(
    """
    Write a number below on the canvas, and our algorithm will predict which number it is.
    """
)

# Upload image
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])
 

#### Prediction ####
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))
    image = np.array(image) #/ 255.0 
    prediction = show_prediction(image, W1, b1, W2, b2)
    st.write('Your image is:', prediction)
    st.image(image, caption='Uploaded Image',use_column_width=True)
