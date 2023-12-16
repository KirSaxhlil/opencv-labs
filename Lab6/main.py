import time

import numpy as np
import matplotlib.pyplot as plt

from keras import Input, Model
from keras.src.layers import Activation, Dense, Dropout, Flatten
from keras.datasets import mnist
from keras.models import load_model
from keras.src.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28,28, 1))
test_images = test_images.reshape((10000, 28,28,1))

train_images = train_images.astype('float32')/255
test_images = test_images.astype('float32')/255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

epochs_list = [5, 10, 15, 20, 25, 30]
results = []

for epochs in epochs_list:
    a = Input(shape=(28,28,1))
    b = Flatten(input_shape=(28,28,1))(a)
    b = Dense(128, activation='relu')(b)
    b = Dense(64, activation='relu')(b)
    b = Dense(10, activation='softmax')(b)
    b = Dropout(0.2)(b)

    model = Model(inputs=a, outputs=b)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    start_time = time.time()

    history = model.fit(train_images, train_labels, epochs=epochs, batch_size=64, validation_split=0.2)

    end_time = time.time()

    training_time = end_time - start_time

    results.append((epochs, round(training_time, 4)))

    model.save(f'models/mlp_model{epochs}.keras')

for epochs,training_time in results:
    print(f"Эпох: {epochs} "
          f"Скорость обучения:{training_time}")

results=[]
for epochs in epochs_list:
    model = load_model(f'models/mlp_model{epochs}.keras')

    _,test_accuracy = model.evaluate(test_images, test_labels)

    start_time = time.time()
    predictions = model.predict(test_images)
    end_time = time.time()
    work_time=end_time-start_time

    results.append((epochs, round(test_accuracy*100,2),round(work_time,4)))

    index = np.random.randint(0, len(test_images))

    plt.imshow(test_images[index].reshape(28, 28), cmap='gray')
    plt.title(f'Предсказание модели: {np.argmax(predictions[index])}')
    plt.show()

for epochs, accuracy,work_time in results:
    print(f"Эпох: {epochs}, Процент корректной работы на тестовых данных: {accuracy}%, "
          f" Скорость работы сети:{work_time}")
