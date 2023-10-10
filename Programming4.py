import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plot
import tensorflow as tf

SIZE = 28
CHANNEL = 1
LAT_DIM = 128
CLASSES = ['T-Shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
EPOCHS = 10
NOISE = 0.5
BATCH_SIZE = 128

dataset = tf.keras.datasets.fashion_mnist
(train_input, train_output), (test_input, test_output) = dataset.load_data()



#Preprocessing
train_input = test_input / 255.0

test_input = test_input / 255.0

# Step 1 Architecture

input_scope = Input(shape=(SIZE, SIZE, CHANNEL))
layer = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_scope)
layer = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(layer)
layer = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(layer)
layer = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(layer)
layer = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(layer)
layer = tf.keras.layers.UpSampling2D((2, 2))(layer)
layer = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(layer)
layer = tf.keras.layers.UpSampling2D((2, 2))(layer)
layer = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(layer)
cae = Model(input_scope, layer)
cae.summary()

# Step 2 Processing
train_input = np.reshape(train_input, (len(train_input), SIZE, SIZE, CHANNEL))
test_input = np.reshape(test_input, (len(test_input), SIZE, SIZE, CHANNEL))
cae.compile(optimizer='adam', loss='binary_crossentropy')

step2_proc = cae.fit(train_input,
                          train_input,
                          epochs=EPOCHS,
                          batch_size=LAT_DIM,
                          shuffle=True,
                          validation_data=(test_input, test_input))

training_loss = step2_proc.history['loss']
testing_loss = step2_proc.history['val_loss']

plot.figure(figsize=(10, 10))
plot.subplot(2, 1, 2)
plot.plot(training_loss, label= 'Training')
plot.plot(testing_loss, label='Testing')
plot.legend(loc='upper right')
plot.ylabel('Loss')
plot.ylim([0,1.0])
plot.title('Training and Testing Loss')
plot.xlabel('Epoch')
plot.show()  

predictions = cae.predict(test_input)

n = 10
plot.figure(figsize=(10, 10))
for i in range(n):
    image = plot.subplot(5, 5, i + 1)
    image.set_title("noisy image")
    plot.imshow(test_input[i].reshape(SIZE, SIZE))
    plot.gray()
    image.get_xaxis().set_visible(False)
    image.get_yaxis().set_visible(False)
    image = plot.subplot(5, 5, i + 1 + n)
    image.set_title("predicted")
    plot.imshow(predictions[i].reshape(SIZE, SIZE))
    plot.gray()
    image.get_xaxis().set_visible(False)
    image.get_yaxis().set_visible(False)    
    
# Step 3 Processing with Noicy dataset
train_input_noise = train_input + NOISE * np.random.normal(loc=0.0, scale=1.0, size=train_input.shape) 
test_input_noice = test_input + NOISE * np.random.normal(loc=0.0, scale=1.0, size=test_input.shape) 
train_input_noise = np.clip(train_input_noise, 0., 1.)
test_input_noice = np.clip(test_input_noice, 0., 1.)



result = cae.fit(train_input_noise, train_input,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True,
                validation_data=(test_input_noice, test_input))

training_loss = result.history['loss']
testing_loss = result.history['val_loss']

plot.figure(figsize=(10, 10))
plot.subplot(2, 1, 2)
plot.plot(training_loss, label= 'Training')
plot.plot(testing_loss, label='Testing')
plot.legend(loc='upper right')
plot.ylabel('Loss')
plot.ylim([0,1.0])
plot.title('Training and Testing Loss')
plot.xlabel('Epoch')
plot.show()  

predictions = cae.predict(test_input_noice)

n = 10
plot.figure(figsize=(10, 10))
for i in range(n):
    image = plot.subplot(5, 5, i + 1)
    image.set_title("noisy image")
    plot.imshow(test_input_noice[i].reshape(SIZE, SIZE))
    plot.gray()
    image.get_xaxis().set_visible(False)
    image.get_yaxis().set_visible(False)
    image = plot.subplot(5, 5, i + 1 + n)
    image.set_title("predicted")
    plot.imshow(predictions[i].reshape(SIZE, SIZE))
    plot.gray()
    image.get_xaxis().set_visible(False)
    image.get_yaxis().set_visible(False)    

#step 4
PROCESSING_SIZE = 1000
prediction_set = np.reshape(predictions, (len(predictions),784))[0:PROCESSING_SIZE, :]
tsne = TSNE()
tsne_transform = tsne.fit_transform(prediction_set)
tsne_transform = np.append(tsne_transform, np.reshape(test_output,(len(test_output),1))[0:PROCESSING_SIZE, :], 1)
tsne_dataframe = pd.DataFrame(data = tsne_transform,
                        columns = ["x",
                                   "y", "output"])

plot.figure(figsize=(16,10))
sns.scatterplot(
    x="x", y="y",
    hue = "output",
    hue_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    palette=sns.color_palette("hls", 10),
    data=tsne_dataframe,
    legend="full",
    alpha=0.3
)
