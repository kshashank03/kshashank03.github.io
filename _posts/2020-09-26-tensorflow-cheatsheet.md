---
layout: post
title: Tensorflow Cheatsheet
---

# Tensorflow 2.x Cheatsheet

# Artificial Neural Networks and Tensorflow Basics

### Build (Compile and Train) a Multi-Layer Sequential Model

This is a regression model, if you wanted a classifier you'd need to have an activation function at the last layer

```python
from tensorflow import keras
import numpy as np

rooms = np.array(range(1, 11))
price = np.array(range(100, 600, 50))

model = keras.models.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dense(units=64) 
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(rooms, price, epochs=100)

print(model.predict([7.0]))
```

### Build a Binary Classifier

```python
train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)
train_generator = train_datagen.flow_from_directory(
    filepath,
    target_size=input_shape,
    batch_size=10,
    **class_mode='binary' #Switch to 'categorical' for multi-class**
)

# Build Model
model = keras.Sequential(layers=[
    keras.layers.Conv2D(filters=64, kernel_size=2, strides=4, input_shape=input_shape, activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Conv2D(filters=32, kernel_size=2, strides=2, activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    **keras.layers.Dense(2, activation='sigmoid')**
])
```

### Build a Multi-class Classifier

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TRAINING_DIR = "/tmp/rps/"
training_datagen = ImageDataGenerator(
      rescale = 1./255,
	    rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

VALIDATION_DIR = "/tmp/rps-test-set/"
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
	**class_mode='categorical'**,
  batch_size=126
)

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(150,150),
	**class_mode='categorical'**,
  batch_size=126
)

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    **tf.keras.layers.Dense(3, activation='softmax')**
])

model.summary()

model.compile(**loss = 'categorical_crossentropy'**, optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(train_generator, epochs=25, steps_per_epoch=20, validation_data = validation_generator, verbose = 1, validation_steps=3)
```

### Using Callbacks to stop training early

First create your callback class and then instantiate it

```python
desired_accuracy = 0.999

class MyCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > desired_accuracy:
            print(f"You have reached the desired accuracy | {desired_accuracy * 100}%")
            self.model.stop_training = True

callbacks = MyCallback()
```

Next add the callbacks object to you `model.fit()` method

```python
history = model.fit(X_train, y_train, epochs=40, **callbacks=[callbacks]**)
```

### Using Callbacks to Save Models at Intervals

```python
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        self.model.save(pwd + f"/imdb_models/model_{epoch}.h5", save_format='h5')
```

### Batch Loading of Data

```python
history = model.fit(X_train, y_train, epochs=40, callbacks=[callbacks]**, batch_size=32**)
```

### Use pretrained models (Transfer Learning)

This is an example of using the Inception Model which performs very well on the ImageNet dataset

We first download the weights for the model from the googleapi URL provided

We then download the skeleton of the model (`model = keras.Sequential([...])`)

We set `include_top=False` because the Inception model has a fully connected layer at the top and we want just the convolutions

We then load the weights into the model and then iterate through the layers to make sure they aren't trainable for when we train the Dense layers we'll be adding to the end of this model

```python
pwd = os.getcwd()

local_weights_file = pwd + '/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

wget.download("https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5",
              local_weights_file)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
  layer.trainable = False
  
# pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output
```

### Extract Features from pretrained models

In the above example we liked the convolutions up until the "mixed7" layer. We used the `last_output` variable to store the shape of that layer and below we're going to use the below code to add our Dense layers and output of the model.

```python
from tensorflow.keras.optimizers import RMSprop

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)                  
# Add a final sigmoid layer for classification
x = layers.Dense  (1, activation='sigmoid')(x)           

model = Model(pre_trained_model.input, x) 

model.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])
```

---

# Inputs and Preprocessing

## General

### Use datasets From tensorflow_datasets (tfds)

**Rock Paper Scissors**

```python
train_data, test_data = tfds.load(name="rock_paper_scissors", split=["train", "test"], batch_size=-1, as_supervised=True)cd

train_examples, train_labels = tfds.as_numpy(train_data)
test_examples, test_labels = tfds.as_numpy(test_data)

datagen = keras.preprocessing.image.ImageDataGenerator(
      rescale = 1./255,
	    rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest',
)

datagen.fit(train_examples)
```

**MNIST**

```python
train, test = keras.datasets.mnist.load_data()

train_images, train_labels = train
train_images = train_images / 255.0
train_images = train_images.reshape(60000, 28, 28, 1)

test_images, test_labels = test
test_images = test_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
```

**Fashion MNIST**

```python
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np

train_data, test_data = tfds.load(name="fashion_mnist", split=["train", "test"], batch_size=-1, as_supervised=True)

train_examples, train_labels = tfds.as_numpy(train_data)
test_examples, test_labels = tfds.as_numpy(test_data)

train_examples = train_examples/255.0
test_examples = test_examples/255.0
```

**IMDB**

```python
train_data, test_data = tfds.load(name="imdb", split=["train", "test"], batch_size=-1, as_supervised=True)

train_examples, train_labels = tfds.as_numpy(train_data)
test_examples, test_labels = tfds.as_numpy(test_data)
```

**Malaria**

```python
train_data = tfds.load(name="malaria", split="train", batch_size=-1, as_supervised=True)

train_examples, train_labels = tfds.as_numpy(train_data)

small_factor = 2000 #This dataset is too large to process all at once, so we only are going to use a part of it

test_size = 0.2

small_train_examples = train_examples[:small_factor]
small_train_labels = train_labels[:small_factor]
small_test_examples = train_examples[small_factor:small_factor + int(small_factor * test_size)]
small_test_labels = train_labels[small_factor:small_factor + int(small_factor * test_size)]

datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
)

datagen.fit(small_train_examples)
```

**yelp_polarity_reviews**

```python
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import re

pwd = os.getcwd()

train_data, test_data = tfds.load(name="yelp_polarity_reviews", split=["train", "test"], batch_size=-1,
                                  as_supervised=True, )

train_examples, train_labels = tfds.as_numpy(train_data)
test_examples, test_labels = tfds.as_numpy(test_data)

train_examples_list = [str(x) for x in train_examples]

test_examples_list = [str(x) for x in test_examples]

vocab_size = 10000  # How many words to learn
embedding_dim = 8  # Length of 2nd dimension of embeddings
max_length = int(max([len(re.findall("[\s]+", str(i))) for i in
                      train_examples]))  # Longest sequence in the training corpus #The longest sequence to allow
trunc_type = 'post'  # If sequence is longer, then trunc pre or post
padding_type = 'post'  # If sequence is shorter, then pad pre or post
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size,
                      oov_token=oov_tok)  # By default the Tokenizer method will lowercase all the letters
tokenizer.fit_on_texts(train_examples_list)
word_index = tokenizer.word_index  # Shows the index assigned to each word
train_sequences = tokenizer.texts_to_sequences(train_examples_list)  # Encodes each sequence

test_sequences = tokenizer.texts_to_sequences(test_examples_list)  # Encodes each sequence

train_padded = pad_sequences(train_sequences, maxlen=max_length, truncating=trunc_type, padding=padding_type,)

test_padded = pad_sequences(test_sequences, maxlen=max_length, truncating=trunc_type, padding=padding_type,)
```

### Use data from CSVs and JSONs

**JSON - Creates a dictionary of each row**

```python
import json

with open("/tmp/sarcasm.json", 'r') as f:
    datastore = json.load(f)
```

**CSV**

I don't see why I can't just use Pandas

```python
import csv
sentences = []
labels = []
with open("/tmp/bbc-text.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
				labels.append(row[0])
				sentences.append(row[1])
```

## Image Classification

### Convert Greyscale Numpy Image to RGB (Useful for Transfer Learning)

```python
train_examples = np.squeeze(np.stack((train_examples,)*3, axis=-1))
test_examples = np.squeeze(np.stack((test_examples,)*3, axis=-1))
```

### Input data in the right shape

You can force a size for images using the ImageDataGenerator class. Look at the bold red section

```python
# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        **target_size=(150, 150),**  # All images will be resized to 150x150
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary' **#Switch to 'categorical' for multi-class**)

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        **target_size=(150, 150)**,
        batch_size=20,
        class_mode='binary' **#Switch to 'categorical' for multi-class**)
```

### Use ImageDataGenerator to import data

```python
pwd = os.getcwd()

filepath = pwd + "/happy-or-sad/"

train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)

train_generator = train_datagen.flow_from_directory(
    filepath,
    target_size=input_shape,
    batch_size=10,
    class_mode='binary' **#Switch to 'categorical' for multi-class**
		validation_split=0.2 #Use to define how much you want to leave for validation
)

TEST_SIZE = 0.2

datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    validation_split=TEST_SIZE #The amount of data to leave behind for testing
)

train_generator = datagen.flow_from_directory(
    filepath,
    target_size=(64, 64),
    batch_size=10,
    class_mode='binary', #Switch to 'categorical' for multi-class
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    filepath,
    target_size=(64, 64),
    batch_size=10,
    class_mode='binary', #Switch to 'categorical' for multi-class
    subset='validation'
)
```

## Natural Language Processing

### Parameters to define for NLP

```python
vocab_size = 10000 #How many words to learn
embedding_dim = 16 #Length of 2nd dimension of embeddings
max_length = 100 #The longest sequence to allow
trunc_type='post' #If sequence is longer, then trunc pre or post
padding_type='post' #If sequence is shorter, then pad pre or post
oov_tok = "<OOV>"
```

### Tokenizing Text for NLP

```python
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'i love my dog',
    'I, love my cat',
    'You love my dog!'
]

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok) #By default the Tokenizer method will lowercase all the letters
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index #Shows the index assigned to each word
sequences = tokenizer.texts_to_sequences(sentences) #Encodes each sequence
```

### Padding Text to Ensure Consistent Length

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

padded = pad_sequences(sequences, maxlen=max_length)
```

## Time Series Forecasting

### Creating Data for Time Series Forecasting

**Trend - Noise - Seasonal Pattern and Seasonality - Necessary Values**

```python
def trend(time, slope=0):
    return slope * time

def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
series = trend(time, 0.1)  
baseline = 10
amplitude = 40
slope = 0.01
noise_level = 2

# Create the series
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
# Update with noise
series += noise(time, noise_level, seed=42)
```

### Windowing Data and Labels for Time Series Models

```python
window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
	dataset = tf.expand_dims(series, axis=-1)
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset

dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
```

---

# Improve Model Performance

### Prevent overfitting using Augmentation and Dropout

**Augmentation**

Look at the bolded red section for the specific transformations that we can implement to reduce overfitting

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(
      rescale=1./255,
      **rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')**

validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        '/tmp/horse-or-human/',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 150x150
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary' **#Switch to 'categorical' for multi-class**)

# Flow training images in batches of 128 using train_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        '/tmp/validation-horse-or-human/',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 150x150
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary' **#Switch to 'categorical' for multi-class**)
```

**Dropout**

We can use Dropout layers in order to remove a random number of neurons from any specified layer. This is helpful to reduce overfitting because neurons near each other tend to have similar weights

```python
keras.Sequential([
keras.layers.Flatten()(last_output)

, keras.layers.Dense(1024, activation='relu')(x)

**, keras.layers.Dropout(0.2)(x)**                  

, keras.layers.Dense  (1, activation='sigmoid')(x)
])
```

### Using Conv2D and MaxPooling layers to improve model performance

The bold section is for binary classifiers only, you'll need to change this for multi-class classifiers

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense**(1, activation='sigmoid')**
])

model.compile(optimizer=RMSprop(lr=0.001), **loss='binary_crossentropy'**, metrics=['accuracy'])
```

### Using Embeddings to Improve NLP Models

Often words can be associated with a given class in an NLP classification task. In this case we can use Embeddings to capture these relations. Below we using the Embedding method to create a  2-D array that we need to flatten. We can use the Flatten method but we might want to use the GlobalAveragePooling1D method instead. 

```python
model = tf.keras.Sequential([
    **tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),**
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
```

### Using LSTMs to Improve NLP Model Performance

```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)), #Return sequences allows you to input another LSTM layer
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

### Using CNNs to Improve NLP Model Performance

```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv1D(128, 7, activation='relu', strides=3),
    tf.keras.layers.Conv1D(128, 7, activation='relu', strides=3),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

### Using GRUs (Gated Recurrent Units) to Improve NLP Model Performance

```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

### Dynamically Changing the Learning Rate in a Time Series Model

```python
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])

plt.semilogx(history.history["lr"], history.history["loss"])
```

![Tensorflow%202%20x%20Cheatsheet%20199bcd691e9d4b62af4e03e435c5d766/Untitled.png](Tensorflow%202%20x%20Cheatsheet%20199bcd691e9d4b62af4e03e435c5d766/Untitled.png)

Check this graph and see that E-6 to E-5 is the optimal rate, change lr in SGD to that

---

# Implementation

### Predict the Next Word in a Sequence (NLP)

```python
## Preprocess Data ##

input_sequences = []
for line in corpus:
	token_list = tokenizer.texts_to_sequences([line])[0]
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		input_sequences.append(n_gram_sequence)

# pad sequences 
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# create predictors and label
xs, labels = input_sequences[:,:-1],input_sequences[:,-1]

ys = tf.keras.utils.to_categorical(labels, num_classes=total_words) #One-hot-encodes all the sequences

## Build Model ##

model = Sequential()
  model.add(Embedding(total_words, 64, input_length=max_sequence_len-1))
  model.add(Bidirectional(LSTM(20)))
  model.add(Dense(total_words, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  history = model.fit(xs, ys, epochs=500, verbose=1)

## Predict ##

seed_text = "Laurence went to dublin"
next_words = 100
  
for _ in range(next_words):
	token_list = tokenizer.texts_to_sequences([seed_text])[0]
	token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
	predicted = model.predict_classes(token_list, verbose=0)
	output_word = ""
	for word, index in tokenizer.word_index.items():
		if index == predicted:
			output_word = word
			break
	seed_text += " " + output_word
print(seed_text)
```

![Tensorflow%202%20x%20Cheatsheet%20199bcd691e9d4b62af4e03e435c5d766/Untitled%201.png](Tensorflow%202%20x%20Cheatsheet%20199bcd691e9d4b62af4e03e435c5d766/Untitled%201.png)

The first half of the code above creates this dataset for prediction

# Evaluate Model Performance

### Predict Results Using a Model

**Regression:**

```python
print(model.predict([7.0]))
```

**Binary Classification**

```python
np.argmax(model.predict(x), axis=-1)
```

**Multi-class Classification:**

```python
model.predict(x) > 0.5).astype("int32")
```

### Plot Loss and Accuracy of the Model

```python
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")

plt.title('Training and validation loss')
```

### Visualize Word Embeddings Using [https://projector.tensorflow.org/](https://projector.tensorflow.org/)

```python
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_sentence(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

import io

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()
```

# Examples

## DNN

## Image Classification

### Fashion MNIST (Same w/ MNIST) (Transfer Learning)

```python
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import Model
import os
import wget
from tensorflow.keras.applications.inception_v3 import InceptionV3

pwd = os.getcwd()

local_weights_file = pwd + '/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

wget.download("https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5",
              local_weights_file)

pre_trained_model = InceptionV3(input_shape = (75, 75, 3),
                                include_top = False,
                                weights = None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
  layer.trainable = False

train_data, test_data = tfds.load(name="fashion_mnist", split=["train", "test"], batch_size=-1, as_supervised=True)

train_examples, train_labels = tfds.as_numpy(train_data)
test_examples, test_labels = tfds.as_numpy(test_data)

train_examples = train_examples / 255.0
test_examples = test_examples / 255.0

train_examples = np.squeeze(np.stack((train_examples,)*3, axis=-1))
test_examples = np.squeeze(np.stack((test_examples,)*3, axis=-1))

model = tf.keras.models.Sequential([
    keras.layers.Input(shape=(28, 28, 3)),
    keras.layers.Lambda(lambda image: tf.image.resize(image, (75, 75))),
    pre_trained_model,
    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

class MyCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_accuracy') > 0.95:
            print(f"You have reached the desired accuracy | {0.95 * 100}%")
            self.model.stop_training = True

callbacks = MyCallback()

history = model.fit(train_examples, train_labels, epochs=10, callbacks=[callbacks], batch_size=64, validation_data=(test_examples, test_labels))
```

### Malaria Prediction

```python
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import os

pwd = os.getcwd()

train_data = tfds.load(name="malaria", split="train", batch_size=-1, as_supervised=True)

train_examples, train_labels = tfds.as_numpy(train_data)

small_train_examples = train_examples[:small_factor]
small_train_labels = train_labels[:small_factor]
small_test_examples = train_examples[small_factor:small_factor + int(small_factor * test_size)]
small_test_labels = train_labels[small_factor:small_factor + int(small_factor * test_size)]

datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
)

datagen.fit(small_train_examples)

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Input(shape=(385, 394, 3)),
    tf.keras.layers.Lambda(lambda image: tf.image.resize(image, (64, 64))),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        self.model.save(pwd + f"/malaria_image_classification_models/model_{epoch}.h5", save_format='h5')

callback = CustomCallback()

history = model.fit(datagen.flow(small_train_examples, small_train_labels, batch_size=64), epochs=25,
                    validation_data=datagen.flow(small_test_examples, small_test_labels), callbacks=[callback])
```

## Natural Language Processing

### Yelp Polarity Reviews

```python
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import re

pwd = os.getcwd()

train_data, test_data = tfds.load(name="yelp_polarity_reviews", split=["train", "test"], batch_size=-1,
                                  as_supervised=True, )

train_examples, train_labels = tfds.as_numpy(train_data)
test_examples, test_labels = tfds.as_numpy(test_data)

train_examples_list = [str(x) for x in train_examples]

test_examples_list = [str(x) for x in test_examples]

vocab_size = 10000  # How many words to learn
embedding_dim = 8  # Length of 2nd dimension of embeddings
max_length = int(max([len(re.findall("[\s]+", str(i))) for i in
                      train_examples]))  # Longest sequence in the training corpus #The longest sequence to allow
trunc_type = 'post'  # If sequence is longer, then trunc pre or post
padding_type = 'post'  # If sequence is shorter, then pad pre or post
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size,
                      oov_token=oov_tok)  # By default the Tokenizer method will lowercase all the letters
tokenizer.fit_on_texts(train_examples_list)
word_index = tokenizer.word_index  # Shows the index assigned to each word
train_sequences = tokenizer.texts_to_sequences(train_examples_list)  # Encodes each sequence

test_sequences = tokenizer.texts_to_sequences(test_examples_list)  # Encodes each sequence

train_padded = pad_sequences(train_sequences, maxlen=max_length, truncating=trunc_type, padding=padding_type,)

test_padded = pad_sequences(test_sequences, maxlen=max_length, truncating=trunc_type, padding=padding_type,)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv1D(128, 7, activation='relu', strides=3),
    tf.keras.layers.Conv1D(128, 7, activation='relu', strides=3),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        self.model.save(pwd + f"/yelp_polarity_reviews_nlp_models/model_{epoch}.h5", save_format='h5')

callbacks = CustomCallback()

history = model.fit(train_padded, y=train_labels, epochs=10, validation_data=(test_padded, test_labels),
                    callbacks=[callbacks], batch_size=128)
```

### Predicting Shakespeare's Next Words:

```python
model = Sequential([
                    tf.keras.layers.Embedding(total_words, 100, input_length=max_sequence_len-1,),
                    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
                    tf.keras.layers.Dense(24, activation='relu',
    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5)),
    tf.keras.layers.Dense(total_words, activation='softmax')
])

# Pick an optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
```

### Wikipedia Toxicity NLP Classification

```python
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

train_data, test_data = tfds.load(name="wikipedia_toxicity_subtypes", split=["train", "test"], batch_size=-1,
                                  as_supervised=True)

train_examples, train_labels = tfds.as_numpy(train_data)
test_examples, test_labels = tfds.as_numpy(test_data)

train_examples_list = [str(x) for x in train_examples]

test_examples_list = [str(x) for x in test_examples]

vocab_size = 10000  # How many words to learn
embedding_dim = 16  # Length of 2nd dimension of embeddings
max_length = 1000  # The longest sequence to allow
trunc_type = 'post'  # If sequence is longer, then trunc pre or post
padding_type = 'post'  # If sequence is shorter, then pad pre or post
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size,
                      oov_token=oov_tok)  # By default the Tokenizer method will lowercase all the letters
tokenizer.fit_on_texts(train_examples_list)
train_sequences = tokenizer.texts_to_sequences(train_examples_list)  # Encodes each sequence
test_sequences = tokenizer.texts_to_sequences(test_examples_list)

train_padded = pad_sequences(train_sequences, maxlen=max_length)
test_padded = pad_sequences(test_sequences, maxlen=max_length)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Conv1D(128, 5, activation='relu', strides=3),
    tf.keras.layers.Conv1D(128, 7, activation='relu', strides=3),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, return_sequences=True)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

history = model.fit(train_padded, y=train_labels, epochs=25, validation_data=(test_padded, test_labels), batch_size=128)
```

## Time Series, Sequences, Predictions

### Predict Stock Prices

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error

pwd = os.getcwd()
TRAIN_SPLIT = 0.8

data_import = pd.read_csv(pwd + "/datasets/Google_Stock_Price_Train.csv")

dataset = data_import['Open']

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(np.array(dataset).reshape(-1, 1))

training_size = int(len(dataset) * TRAIN_SPLIT)
test_size = len(dataset) - training_size
train_dataset, test_dataset = dataset[0:training_size, :], dataset[training_size:, :]

def create_dataset(data, time_step=1):
    dataX, dataY = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]  # i=0, 0,1,2,3-----99   100
        dataX.append(a)
        dataY.append(data[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 15
X_train, y_train = create_dataset(train_dataset, time_step)
X_test, y_test = create_dataset(test_dataset, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model = keras.models.Sequential([
    keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='causal', activation='relu',
                        input_shape=(time_step, 1)),
    keras.layers.Dropout(0.5),
    keras.layers.LSTM(50),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1)
])

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=16, verbose=1)

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

print(math.sqrt(mean_squared_error(scaler.inverse_transform(y_train.reshape(-1, 1)), train_predict)))

print(math.sqrt(mean_squared_error(y_test.reshape(-1, 1), test_predict)))

plt.plot(scaler.inverse_transform(dataset))
plt.plot(range(time_step, len(train_predict) + time_step), train_predict, c='b')
plt.plot(range(len(train_predict) + time_step, len(train_predict) + len(test_predict) + time_step), test_predict, c='k')
plt.show()
```