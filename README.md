# Neural Network to Recognize Handwritten Numbers

### Update - 30/01/2024
We remade this project utilizing Tensorflow to compare the process and the accuracy with our original neural network. The accuracy improved by about 9 percentage points. I have added all relevant information for this updated version, below.

### About this project
This project was built in collaboration with a classmate. After my team won the coding challenge at a Neural Networks workshop hosted by the IE chapter of Google Developer Student Clubs, our interest in neural networks was sparked. Building off the base code from the workshop, we were able to create a neural network that does not utilize any ML libraries and that can accept a hand-drawn number and predict what number is depicted. We then deployed it to the web. 

### Technologies 
- The original project is coded in Python. Libraries used include: 🐼pandas | 🧮numpy | 📈matplotlib | 🥒pickle | 🎨streamlit | 🖼️PIL
- The updated project is coded in Python. Libraries used include: 🤖tensorflow | 🧮numpy | 📈matplotlib | 🎨streamlit | 🖼️PIL

### Deployed app
- The original deployed app URL is: https://handwrittennumbers.streamlit.app/
- The updated deployed app URL is: IN PROGRESS

### Files
For the original project:
- neural.py: The script for the deployed app
- trained_params.pkl: The file holding the trained parameters of the neural network
- .streamlit: This folder holds a configuration file for the streamlit app
- training model.ipynb: The code for training the neural network; this code produces the trained_params.pkl file
- requirements.txt: The version requirements file for deployment in streamlit
- Usage_Recording.mp4: Recording of example usage of this app

For the updated project:
- neural_tensorflow.py: The script for the deployed app
- training_model_tensorflow.ipynb: The code for training the neural network; this code produces the trained_params_tensorflow folder
- trained_params_tensorflow: The saved trained model

