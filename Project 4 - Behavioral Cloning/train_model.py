import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from model import Model


def load_images(driving_log):
    images = []
    for index, line in driving_log.iterrows():
        center = cv2.imread(line['center'])
        left = cv2.imread(line['right'])
        right = cv2.imread(line['left'])
        images.append([center, left, right])
    return np.array(images), driving_log


def train_test(images, driving_log_images):
    # y = driving_log_images[['steering', 'throttle', 'brake', 'speed']].to_numpy()
    y = driving_log_images['steering'].to_numpy()
    return train_test_split(images, y, test_size=0.25, random_state=42)


def get_train_test_data(driving_log_path):
    driving_log = pd.read_csv(driving_log_path, sep=',')
    images, driving_log_images = load_images(driving_log)
    return train_test(images, driving_log_images)


if __name__ == '__main__':
    lenet = Model()
    X_train, X_test, y_train, y_test = get_train_test_data('./recorded data/Track 1 Lap 1/driving_log.csv')
    history = lenet.train(X_train, y_train)
    scores = lenet.test(X_test, y_test)
    lenet.save('model.h5')
