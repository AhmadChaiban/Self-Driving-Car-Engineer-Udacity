import tensorflow as tf
from sklearn.metrics import r2_score

class Model:
    def __init__(self):
        self.n_classes = 1
        self.LeNet = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)),

            tf.keras.layers.Cropping2D(cropping=((50, 20), (0, 0))),

            tf.keras.layers.Conv2D(filters=24, kernel_size=(5, 5), activation='relu', strides=(1, 1), padding='valid'),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(filters=36, kernel_size=(5, 5), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(filters=48, kernel_size=(5, 5), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(120, activation='relu', kernel_regularizer='l2'),
            tf.keras.layers.Dense(84, activation='relu'),
            tf.keras.layers.Dense(self.n_classes, activation='sigmoid')
        ])

        self.inception = self.prepare_inception()

    def prepare_inception(self):
        inception_model = tf.keras.applications.InceptionV3(
            include_top=True, weights='imagenet', input_tensor=None, input_shape=None,
            pooling=None, classes=1000, classifier_activation=None
        )
        for layer in inception_model.layers:
            layer.trainable = False
        # Flatten the output layer to 1 dimension
        x = tf.keras.layers.Flatten()(inception_model.output)
        # Add a fully connected layer with 1,024 hidden units and ReLU activation
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        # Add a dropout rate of 0.2
        x = tf.keras.layers.Dropout(0.2)(x)
        # Add a final sigmoid layer for classification
        x = tf.keras.layers.Dense(self.n_classes, activation='sigmoid')(x)
        return tf.keras.Model(inception_model.input, x)

    def train(self, X_train, y_train):
        epochs = 5
        batch_size = 32
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, name='SGD')
        loss_function = tf.keras.losses.log_cosh
        self.LeNet.compile(optimizer=optimizer, loss=loss_function, metrics=['mae'])
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
