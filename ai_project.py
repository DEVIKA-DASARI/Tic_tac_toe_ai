import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, LSTM, Embedding, Input, Dropout, add
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# Load pre-trained ResNet model
resnet = ResNet50(include_top=True, weights="imagenet")

# Remove classification layer
resnet = Model(inputs=resnet.input, outputs=resnet.layers[-2].output)


# Function to preprocess images
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img


# Function to encode image into features
def encode_image(image):
    image = preprocess_image(image)
    feature_vector = resnet.predict(image)
    feature_vector = np.reshape(feature_vector, feature_vector.shape[1])
    return feature_vector


# Maximum sequence length
max_length = 40

# Load model
embedding_size = 300
units = 256
encoder_dim = 2048

# Image feature extractor model
inputs1 = Input(shape=(encoder_dim,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(units, activation="relu")(fe1)

# Caption sequence model
inputs2 = Input(shape=(max_length,))
se1 = Embedding(input_dim=10000, output_dim=embedding_size, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(units)(se2)

# Merging both models
decoder1 = add([fe2, se3])
decoder2 = Dense(units, activation="relu")(decoder1)
outputs = Dense(10000, activation="softmax")(decoder2)

# Combined model
model = Model(inputs=[inputs1, inputs2], outputs=outputs)

# Load trained weights
model.load_weights("model_weights.h5")


# Function to generate caption
def generate_caption(image_path):
    photo = encode_image(image_path)
    in_text = "<start>"
    for i in range(max_length):
        sequence = tf.convert_to_tensor([[word_index[in_text]]])
        yhat = model.predict([photo.reshape(1, -1), sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = index_word[yhat]
        in_text += " " + word
        if word == "<end>":
            break
    caption = in_text.split()
    caption = caption[1:-1]
    caption = " ".join(caption)
    return caption


# Example usage
dataset_folder = "dataset"
captions_file = "captions.txt"

# Read captions from the text file
captions = {}
with open(os.path.join(dataset_folder, captions_file), "r") as file:
    for line in file:
        image_path, caption = line.strip().split(",")
        captions[image_path.strip()] = caption.strip()

# Iterate over the dataset folder
for image_file in os.listdir(dataset_folder):
    if image_file.endswith(".jpg") or image_file.endswith(".png"):
        image_path = os.path.join(dataset_folder, image_file)
        if image_path in captions:
            caption = captions[image_path]

            # Generate caption
            generated_caption = generate_caption(image_path)

            # Print or use the generated caption
            print("Image Path:", image_path)
            print("Original Caption:", caption)
            print("Generated Caption:", generated_caption)
            print()

            # Optionally, plot the image with the generated caption
            img = plt.imread(image_path)
            plt.imshow(img)
            plt.title(generated_caption)
            plt.axis("off")
            plt.show()
