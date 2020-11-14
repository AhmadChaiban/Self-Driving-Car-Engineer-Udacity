import tensorflow as tf
from sklearn.metrics import r2_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Lambda, Cropping2D, Conv2D, MaxPooling2D, Flatten, \
                                    Dense, Dropout, LayerNormalization
import matplotlib.pyplot as plt

class Model:
    def __init__(self, selected_model):
        self.n_classes = 1
        self.LeNet = Sequential([
            Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)),
            Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)),

            Conv2D(filters=24, kernel_size=(5, 5), activation='relu', strides=(1, 1), padding='valid'),
            MaxPooling2D((2, 2)),

            Conv2D(filters=36, kernel_size=(5, 5), activation='relu'),
            MaxPooling2D((2, 2)),

            Conv2D(filters=48, kernel_size=(5, 5), activation='relu'),
            MaxPooling2D((2, 2)),

            Conv2D(filters=64, kernel_size=(1, 1), activation='relu'),
            MaxPooling2D((2, 2)),

            Conv2D(filters=64, kernel_size=(1, 1), activation='relu'),
            MaxPooling2D((2, 2)),

            Flatten(),

            Dense(120, activation='relu', kernel_regularizer='l2'),
            Dense(84, activation='relu'),
            Dense(self.n_classes, activation='sigmoid')
        ])

        self.nvidia = Sequential([
            Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)),
            Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)),

            LayerNormalization(epsilon=0.001),

            Conv2D(filters=24, kernel_size=(5, 5), activation='relu', strides=(3, 3), padding='valid'),
            Conv2D(filters=36, kernel_size=(5, 5), activation='relu'),
            Conv2D(filters=48, kernel_size=(5, 5), activation='relu'),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),

            Flatten(),

            Dense(1164, activation='relu'),
            Dense(100, activation='relu'),
            Dense(50, activation='relu'),
            Dense(10, activation='relu'),
            Dense(self.n_classes, activation='sigmoid')

        ])

        self.inception = self.prepare_inception()

        if selected_model.lower() == 'lenet':
            self.current_model = self.LeNet
        elif selected_model.lower() == 'nvidia' or selected_model.lower() is None:
            self.current_model = self.nvidia
        elif selected_model.lower() == 'inception':
            self.selected_model = self.inception
        else:
            raise Exception('Please select a valid model: LeNet, Nvidia or Inception')

    def prepare_inception(self):
        inception_model = tf.keras.applications.InceptionV3(
            include_top=True, weights='imagenet', input_tensor=None, input_shape=None,
            pooling=None, classes=1000, classifier_activation=None
        )
        for layer in inception_model.layers:
            layer.trainable = False
        # Flatten the output layer to 1 dimension
        x = Flatten()(inception_model.output)
        # Add a fully connected layer with 1,024 hidden units and ReLU activation
        x = Dense(1024, activation='relu')(x)
        # Add a dropout rate of 0.2
        x = Dropout(0.2)(x)
        # Add a final sigmoid layer for classification
        x = Dense(self.n_classes, activation='sigmoid')(x)
        return tf.keras.Model(inception_model.input, x)

    def train(self, X_train, y_train):
        epochs = 3
        batch_size = 32
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, name='SGD')
        loss_function = tf.keras.losses.log_cosh
        self.current_model.compile(optimizer=optimizer, loss=loss_function, metrics=['mae'])
        history = self.current_model.fit(x=X_train,
                                         y=y_train,
                                         batch_size=batch_size,
                                         epochs=epochs,
                                         validation_split=0.2)
        return history

    def test(self, X_test, y_test):
        y_pred = self.current_model.predict(X_test.astype(dtype='float32'))
        score = r2_score(y_test, y_pred)
        print(f"r2 Score = {score}")
        return score

    def save(self, path):
        self.current_model.save(path)
