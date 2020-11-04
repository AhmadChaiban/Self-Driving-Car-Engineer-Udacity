import tensorflow as tf
from sklearn.metrics import r2_score

class Model:
    def __init__(self):
        self.n_classes = 4
        self.LeNet = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)),
            tf.keras.layers.Cropping2D(cropping=((50, 20), (0,0))),
            tf.keras.layers.Conv2D(filters=6,
                                   kernel_size=(5, 5),
                                   activation='relu',
                                   strides=(1, 1),
                                   padding='valid'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(filters=16,
                                   kernel_size=(5, 5),
                                   activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(120, activation='relu'),
            tf.keras.layers.Dense(84, activation='relu'),
            tf.keras.layers.Dense(self.n_classes, activation='sigmoid')
        ])

    def train(self, X_train, y_train):
        epochs = 1
        batch_size = 512
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, name='Adam')
        rate = 0.1
        loss_function = tf.keras.losses.categorical_crossentropy
        self.LeNet.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
        history = []
        for i in range(0, 3):
            history.append(self.LeNet.fit(x=X_train[:, i],
                                          y=y_train,
                                          batch_size = batch_size,
                                          epochs=epochs,
                                          validation_split=0.2)
                           )
        return history

    def test(self, X_test, y_test):
        scores=[]
        for i in range(0, 3):
            y_pred = self.LeNet.predict(X_test[:, i])
            scores.append(r2_score(y_test, y_pred))
        print(scores)
        return scores

    def save(self, path):
        self.LeNet.save(path)
