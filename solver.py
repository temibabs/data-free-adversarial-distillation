import sys

import tensorflow as tf
from tensorflow_core.python.keras.models import load_model

from data_loader import get_dataset
from models.generator import GeneratorA
from models.student import LeNet5Half
from models.teacher import LeNet5


class Solver:
    DEFAULTS = {}

    def __init__(self, data, conf):
        # Data loader
        self.__dict__.update(Solver.DEFAULTS, **conf)
        self.data = data

        self.student = None
        self.teacher = None
        self.generator = None

        self.logger = None

        if self.use_tensorboard:
            self.build_tensorboard()

        self.build_models()

    def build_tensorboard(self):
        from logger import Logger

        print("[+] Setting up Tensorboard for logging...")
        self.logger = Logger(self.log_path)
        self.logger.data_scalar_summary('input', self.data, 1)

    def build_models(self):
        print('[+] Building the teacher')

        # Build the teacher
        if self.train_teacher:
            self.teacher = LeNet5()

            self.teacher.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                loss=tf.keras.losses.mean_absolute_error,
                metrics=[tf.keras.metrics.mean_absolute_error])

            self.teacher.build((None, 32, 32, 1))
            self.teacher.summary()

            optimizer = tf.keras.optimizers.Adam()
            loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
            train_loss = tf.keras.metrics.Mean()
            train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
                name='train_accuracy')

            print('Training teacher...')

            @tf.function
            def train_step(input_images, labels):
                with tf.GradientTape() as teacher_tape:
                    logits = self.teacher(input_images)
                    logits = tf.squeeze(logits)

                    loss = loss_object(labels, logits)

                gradients = teacher_tape.gradient(
                    loss,
                    self.teacher.trainable_variables)

                optimizer.apply_gradients(
                    zip(gradients, self.teacher.trainable_variables))

                train_loss(loss)
                train_accuracy(labels, logits)

            self.test_data = get_dataset(10, 'test')
            best_accuracy = 0
            for epoch in range(self.num_epochs):
                train_loss.reset_states()
                train_accuracy.reset_states()
                for images, labels in self.data:
                    train_step(images, labels)

                template = 'Epoch {}, Loss: {}, Accuracy: {}'
                print(template.format(epoch + 1,
                                      train_loss.result(),
                                      train_accuracy.result() * 100))

                if best_accuracy < train_accuracy.result()*100:
                    self.teacher.save(self.teacher_model_path)
                    best_accuracy = train_accuracy.result()*100

            print('[+] Model trained.')

        else:
            self.teacher = load_model(self.teacher_model_path, compile=False)

        # Build the student
        self.student = LeNet5Half()

        # Build the generator
        self.generator = GeneratorA()

        gen_optimizer = tf.keras.optimizers.Adam()
        gen_loss_object = tf.keras.losses.MeanAbsoluteError()
        gen_train_loss = tf.keras.metrics.Mean()
        gen_train_accuracy = tf.keras.metrics.CategoricalAccuracy(
             name='train_accuracy')

        @tf.function
        def generator_train_step(noise):
            with tf.GradientTape() as generator_tape:
                gen_train_loss.reset_states()
                gen_train_accuracy.reset_states()

                inputs = self.generator(noise)
                teacher_output = tf.squeeze(self.teacher(inputs))
                student_output = tf.squeeze(self.student(inputs))

                loss = -1 * gen_loss_object(teacher_output, student_output)

            gradients = generator_tape.gradient(
                loss,
                self.generator.trainable_variables)

            gen_optimizer.apply_gradients(
                zip(gradients, self.generator.trainable_variables))

            gen_train_loss(loss)
            gen_train_accuracy(teacher_output, student_output)

        im_optimizer = tf.keras.optimizers.Adam()
        im_loss_object = tf.keras.losses.MeanAbsoluteError()
        im_train_loss = tf.keras.metrics.Mean()
        im_train_accuracy = tf.keras.metrics.CategoricalAccuracy(
            name='train_accuracy')
        @tf.function
        def imitator_train_step(noise):
            with tf.GradientTape() as imitator_tape:
                im_train_loss.reset_states()
                im_train_accuracy.reset_states()

                inputs = self.generator(noise)
                teacher_output = tf.squeeze(self.teacher(inputs))
                student_output = tf.squeeze(self.student(inputs))

                loss = im_loss_object(teacher_output, student_output)

            gradients = imitator_tape.gradient(
                loss,
                self.student.trainable_variables)

            im_optimizer.apply_gradients(
                zip(gradients, self.student.trainable_variables))

            im_train_loss(loss)
            im_train_accuracy(teacher_output, student_output)

        best_accuracy = 0
        for epoch in range(self.num_epochs):
            # train_loss.reset_states()
            # train_accuracy.reset_states()

            # Imitation Stage
            for step in range(self.generator_steps):
                noise = tf.random.normal([self.batch_size, 100],
                                         dtype=tf.float64)
                imitator_train_step(noise)
                template = 'Epoch {}, Step {}, Imitation Loss: {}, Accuracy: {}'
                print(template.format(epoch + 1, step + 1,
                                      im_train_loss.result(),
                                      im_train_accuracy.result() * 100))

            # Generation Stage
            gen_noise = tf.random.normal([self.batch_size, 100],
                                         dtype=tf.float64)
            generator_train_step(gen_noise)

            if best_accuracy < im_train_accuracy.result() * 100:
                self.student.save(self.student_model_path)
                best_accuracy = im_train_accuracy.result() * 100

        print('[+] Training complete.')

    def test(self, test_dataset):
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='test_accuracy')
        self.student(test_dataset)
