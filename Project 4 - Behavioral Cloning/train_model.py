import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from model import Model
import matplotlib.pyplot as plt


def image_setup(path):
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    image_flipped = cv2.flip(image, 1)
    return image, image_flipped


def load_images(driving_log):
    images = []
    steering = []
    for index, line in driving_log.iterrows():

        y = line['steering']

        # center, center_flipped = image_setup('./udacity data/' + line['center'])
        # left, left_flipped = image_setup('./udacity data/' + line['left'].replace(' ', ''))
        # right, right_flipped = image_setup('./udacity data/' + line['right'].replace(' ', ''))

        center, center_flipped = image_setup(line['center'])
        left, left_flipped = image_setup(line['left'])
        right, right_flipped = image_setup(line['right'])

        images.append(center)
        images.append(center_flipped)
        images.append(right)
        images.append(right_flipped)
        images.append(left)
        images.append(left_flipped)

        steering.append(y)
        steering.append(y * -1.0)
        steering.append(y - 0.2)
        steering.append((y - 0.2) * -1.0)
        steering.append(y + 0.2)
        steering.append((y + 0.2) * -1.0)
    return images, steering


def train_test(image_data, steering_data):
    # y = driving_log_images[['steering', 'throttle', 'brake', 'speed']].to_numpy()
    return train_test_split(np.array(image_data), np.array(steering_data), test_size=0.25, random_state=42)


def get_train_test_data(driving_log_path):
    driving_log = pd.read_csv(driving_log_path, sep=',')
    image_data, steering_data = load_images(driving_log)
    return train_test(image_data, steering_data)


if __name__ == '__main__':
    # Choices: LeNet, Nvidia, Inception
    model = Model('nvidia')
    X_train, X_test, y_train, y_test = get_train_test_data('./recorded data/Track 1 Lap 3/driving_log.csv')
    # X_train, X_test, y_train, y_test = get_train_test_data('./udacity data/driving_log.csv')
    history = model.train(X_train, y_train, epochs=3, batch_size=1024)
    scores = model.test(X_test, y_test)
    model.save('model.h5')
