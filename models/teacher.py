import tensorflow as tf


class LeNet5(tf.keras.Model):
    def __init__(self):
        super(LeNet5, self).__init__()

        # LeNet-5
        self.conv1 = tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5),
                                            activation='relu')
        self.max_pool1 = tf.keras.layers.MaxPool2D()
        self.conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5),
                                            activation='relu')
        self.max_pool2 = tf.keras.layers.MaxPool2D()
        self.conv3 = tf.keras.layers.Conv2D(filters=120, kernel_size=(5, 5),
                                            activation='relu')
        self.fc1 = tf.keras.layers.Dense(units=84)
        self.fc2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.max_pool1(x)

        x = self.conv2(x)
        x = self.max_pool2(x)

        x = self.conv3(x)

        x = self.fc1(x)
        return self.fc2(x)
