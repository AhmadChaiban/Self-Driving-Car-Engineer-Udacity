import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from model import Model
import matplotlib.pyplot as plt


def image_setup(path):
    # return cv2.flip(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), 1)
    return cv2.flip(cv2.imread(path), 1)


def load_images(driving_log):
    images = []
    steering = []
    for index, line in driving_log.iterrows():

        y = line['steering']

        center = image_setup(line['center'].replace(' ', ''))
        left = image_setup(line['right'].replace(' ', ''))
        right = image_setup(line['left'].replace(' ', ''))

        images.append(center)
        images.append(right)
        images.append(left)

        steering.append(y)
        steering.append(y - 0.2)
        steering.append(y + 0.2)
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
    X_train, X_test, y_train, y_test = get_train_test_data('./udacity data/driving_log.csv')
    history = model.train(X_train, y_train)
    scores = model.test(X_test, y_test)
    model.save('model.h5')
