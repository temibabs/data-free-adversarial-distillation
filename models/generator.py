import tensorflow as tf


class GeneratorA(tf.keras.Model):
    def __init__(self):
        super(GeneratorA, self).__init__()

        # Generator-A
        self.fc1 = tf.keras.layers.Dense(units=10*10*256)
        self.reshape1 = tf.keras.layers.Reshape(target_shape=(10, 10, 256))
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.upsample1 = tf.keras.layers.UpSampling2D()

        self.conv1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3))
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.lrelu1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.upsample2 = tf.keras.layers.UpSampling2D()

        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3))
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.lrelu2 = tf.keras.layers.LeakyReLU(alpha=0.2)

        self.conv3 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3),
                                            activation='tanh')
        self.batch_norm4 = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.reshape1(x)
        x = self.batch_norm1(x)
        x = self.upsample1(x)

        x = self.conv1(x)
        x = self.batch_norm2(x)
        x = self.lrelu1(x)
        x = self.upsample2(x)

        x = self.conv2(x)
        x = self.batch_norm3(x)
        x = self.lrelu2(x)

        x = self.conv3(x)
        x = self.batch_norm4(x)

        return x
