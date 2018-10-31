import pandas
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import CSVLogger
from keras.datasets import cifar10
from keras.engine.saving import load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class Cifar10CNN:
    def __init__(self):
        self.model = 0
        self.training_history = 0
        self.accuracy = 0
        self.val_accuracy = 0
        self.loss = 0
        self.val_loss = 0
        self.epochs_plot = 0

    def train_CNN(self):
        c10 = cifar10.load_data()
        train_X, train_Y, test_X, test_Y = (b for a in c10 for b in a)

        # Check training and testing data shape
        print('Training data shape : ', train_X.shape, train_Y.shape)
        print('Testing data shape : ', test_X.shape, test_Y.shape)

        # Find the number of unique labels from the training label set
        classes = np.unique(train_Y)
        print('Total number of outputs : ', len(classes))
        print('Output classes : ', classes)

        # Reshape data set into 32x32x3 matrices
        train_X = train_X.reshape(-1, 32, 32, 3)
        test_X = test_X.reshape(-1, 32, 32, 3)
        print("Reshaped training data : ", train_X.shape)
        print("Reshaped test data : ", test_X.shape)

        # Convert data to float32 and normalize it to 0-1 range
        train_X = train_X.astype('float32')
        test_X = test_X.astype('float32')
        train_X = train_X / 255.0
        test_X = test_X / 255.0

        # Change the labels from categorical to one-hot encoding
        train_Y_one_hot = to_categorical(train_Y)
        test_Y_one_hot = to_categorical(test_Y)

        # Display the change for category label using one-hot encoding
        print('Original label : ', train_Y[0])
        print('After conversion to one-hot : ', train_Y_one_hot[0])

        # Partition training data so that 80% is used for training and 20% for validating
        train_X, valid_X, train_label, valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)
        print("Partitioned train data : ", train_X.shape)
        print("Partitioned validation data : ", valid_X.shape)
        print("Partitioned train labels : ", train_label.shape)
        print("Partitioned validation labels : ", valid_label.shape)

        # Define some constants to be used in code
        batch_size = 512  # 64 originally
        epochs = 40  # 20 originally
        num_classes = 10

        for kernel in [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]:
            for stride in [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]:
                log = 'cifar10_ker' + str(kernel) + '_str' + str(stride) + '.log'
                model = 'cifar10_ker' + str(kernel) + '_str' + str(stride) + '.h5py'
                dropout_log = 'cifar10_dropout_ker' + str(kernel) + '_str' + str(stride) + '.log'
                dropout_model = 'cifar10_dropout_ker' + str(kernel) + '_str' + str(stride) + '.h5py'

                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                # # Train without dropout:
                # Create a training model layers
                cifar10_model = Sequential()
                cifar10_model.add(
                        Conv2D(32, kernel_size=kernel, strides=stride, activation='linear', padding='same',
                               input_shape=(32, 32, 3)))
                cifar10_model.add(LeakyReLU(alpha=0.1))
                cifar10_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
                cifar10_model.add(Conv2D(64, kernel_size=kernel, strides=stride, activation='linear', padding='same'))
                cifar10_model.add(LeakyReLU(alpha=0.1))
                cifar10_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
                cifar10_model.add(Conv2D(128, kernel_size=kernel, strides=stride, activation='linear', padding='same'))
                cifar10_model.add(LeakyReLU(alpha=0.1))
                cifar10_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
                cifar10_model.add(Flatten())
                cifar10_model.add(Dense(128, activation='linear'))
                cifar10_model.add(LeakyReLU(alpha=0.1))
                cifar10_model.add(Dense(num_classes, activation='softmax'))

                # Compile model
                cifar10_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                                      metrics=['accuracy'])

                # Show compiled model
                # cifar10_model.summary()

                # Train the model
                csv_logger = CSVLogger(log, separator=',', append=False)
                training_history = cifar10_model.fit(train_X, train_label, batch_size=batch_size, verbose=0, epochs=epochs,
                                                  validation_data=(valid_X, valid_label), callbacks=[csv_logger])
                # Save the model to file
                cifar10_model.save(model)

                # Evaluate the model
                test_eval = cifar10_model.evaluate(test_X, test_Y_one_hot, verbose=1)
                print('Test loss kernel: ' + str(kernel) + ' stride: ' + str(stride) + ': ', test_eval[0])
                print('Test accuracy kernel: ' + str(kernel) + ' stride: ' + str(stride) + ': ', test_eval[1])

                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                # # Train with dropout:
                # Create a training model layers
                cifar10_model = Sequential()
                cifar10_model.add(
                        Conv2D(32, kernel_size=kernel, strides=stride, activation='linear', padding='same',
                               input_shape=(32, 32, 3)))
                cifar10_model.add(LeakyReLU(alpha=0.1))
                cifar10_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
                cifar10_model.add(Dropout(0.5))
                cifar10_model.add(Conv2D(64, kernel_size=kernel, strides=stride, activation='linear', padding='same'))
                cifar10_model.add(LeakyReLU(alpha=0.1))
                cifar10_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
                cifar10_model.add(Dropout(0.25))
                cifar10_model.add(Conv2D(128, kernel_size=kernel, strides=stride, activation='linear', padding='same'))
                cifar10_model.add(LeakyReLU(alpha=0.1))
                cifar10_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
                cifar10_model.add(Dropout(0.4))
                cifar10_model.add(Flatten())
                cifar10_model.add(Dense(128, activation='linear'))
                cifar10_model.add(LeakyReLU(alpha=0.1))
                cifar10_model.add(Dropout(0.3))
                cifar10_model.add(Dense(num_classes, activation='softmax'))

                # Compile model
                cifar10_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                                      metrics=['accuracy'])

                # Show compiled model
                # cifar10_model.summary()

                # Train the model
                csv_logger = CSVLogger(dropout_log, separator=',', append=False)
                training_history = cifar10_model.fit(train_X, train_label, batch_size=batch_size, epochs=epochs, verbose=0,
                                                     validation_data=(valid_X, valid_label), callbacks=[csv_logger])

                # Save the model to file
                cifar10_model.save(dropout_model)

                # Evaluate the model
                test_eval = cifar10_model.evaluate(test_X, test_Y_one_hot, verbose=1)
                print('Test loss kernel: ' + str(kernel) + ' stride: ' + str(stride) + ' dropout: ', test_eval[0])
                print('Test accuracy kernel: ' + str(kernel) + ' stride: ' + str(stride) + ' dropout: ', test_eval[1])

    def train_best(self):
        c10 = cifar10.load_data()
        train_X, train_Y, test_X, test_Y = (b for a in c10 for b in a)

        # Check training and testing data shape
        print('Training data shape : ', train_X.shape, train_Y.shape)
        print('Testing data shape : ', test_X.shape, test_Y.shape)

        # Find the number of unique labels from the training label set
        classes = np.unique(train_Y)
        print('Total number of outputs : ', len(classes))
        print('Output classes : ', classes)

        # Reshape data set into 32x32x3 matrices
        train_X = train_X.reshape(-1, 32, 32, 3)
        test_X = test_X.reshape(-1, 32, 32, 3)
        print("Reshaped training data : ", train_X.shape)
        print("Reshaped test data : ", test_X.shape)

        # Convert data to float32 and normalize it to 0-1 range
        train_X = train_X.astype('float32')
        test_X = test_X.astype('float32')
        train_X = train_X / 255.0
        test_X = test_X / 255.0

        # Change the labels from categorical to one-hot encoding
        train_Y_one_hot = to_categorical(train_Y)
        test_Y_one_hot = to_categorical(test_Y)

        # Display the change for category label using one-hot encoding
        print('Original label : ', train_Y[0])
        print('After conversion to one-hot : ', train_Y_one_hot[0])

        # Partition training data so that 80% is used for training and 20% for validating
        train_X, valid_X, train_label, valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)
        print("Partitioned train data : ", train_X.shape)
        print("Partitioned validation data : ", valid_X.shape)
        print("Partitioned train labels : ", train_label.shape)
        print("Partitioned validation labels : ", valid_label.shape)

        # Define some constants to be used in code
        batch_size = 512  # 64 originally
        epochs = 200  # 20 originally
        num_classes = 10
        kernel = (3,3)
        stride = (1,1)

        # Create a training model layers
        cifar10_model = Sequential()
        cifar10_model.add(
                Conv2D(32, kernel_size=kernel, strides=stride, activation='linear', padding='same',
                       input_shape=(32, 32, 3)))
        cifar10_model.add(LeakyReLU(alpha=0.1))
        cifar10_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        cifar10_model.add(Dropout(0.5))
        cifar10_model.add(Conv2D(64, kernel_size=kernel, strides=stride, activation='linear', padding='same'))
        cifar10_model.add(LeakyReLU(alpha=0.1))
        cifar10_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        cifar10_model.add(Dropout(0.4))
        cifar10_model.add(Conv2D(128, kernel_size=kernel, strides=stride, activation='linear', padding='same'))
        cifar10_model.add(LeakyReLU(alpha=0.1))
        cifar10_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        cifar10_model.add(Dropout(0.4))
        cifar10_model.add(Flatten())
        cifar10_model.add(Dense(128, activation='linear'))
        cifar10_model.add(LeakyReLU(alpha=0.1))
        cifar10_model.add(Dropout(0.3))
        cifar10_model.add(Dense(num_classes, activation='softmax'))

        # Compile model
        cifar10_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                              metrics=['accuracy'])

        # Show compiled model
        cifar10_model.summary()

        # Train the model
        csv_logger = CSVLogger('best.log', separator=',', append=False)
        training_history = cifar10_model.fit(train_X, train_label, batch_size=batch_size, epochs=epochs, verbose=0,
                                             validation_data=(valid_X, valid_label), callbacks=[csv_logger])

        # Save the model to file
        cifar10_model.save('best.h5py')

        # Evaluate the model
        predicted_classes = cifar10_model.predict(test_X)
        test_eval = cifar10_model.evaluate(test_X, test_Y_one_hot, verbose=1)
        print('Test loss kernel: ' + str(kernel) + ' stride: ' + str(stride) + ' dropout: ', test_eval[0])
        print('Test accuracy kernel: ' + str(kernel) + ' stride: ' + str(stride) + ' dropout: ', test_eval[1])

        self.plot_best_plot()

        predicted_classes = np.argmax(np.round(predicted_classes), axis=1)
        target_names = ["Class {}".format(i) for i in range(num_classes)]
        print(classification_report(test_Y, predicted_classes, target_names=target_names))

    def plot_best_plot(self):
        self.training_history = pandas.read_csv('best.log', sep=',', engine='python')
        self.load_history_data()

        # Set names for plot
        fig = 'best'
        loss = 'Loss for kernel: (3, 3) stride: (1, 1) with dropout'
        accuracy = 'Accuracy for kernel: (3, 3) stride: (1, 1) with dropout'

        # Plot
        self.plot_to_file(accuracy, loss, fig)

    def write_plots_to_file(self):
        for kernel in [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]:
            for stride in [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]:
                # Load model and history
                self.load_normal_history_from_file(kernel, stride)

                # Load history data
                self.load_history_data()

                # Set names for plot
                fig_name = 'cifar10_ker' + str(kernel) + '_str' + str(stride)
                loss_title = 'Loss for kernel: ' + str(kernel) + ' stride: ' + str(stride)
                accuracy_title = 'Accuracy for kernel: ' + str(kernel) + ' stride: ' + str(stride)

                # Plot
                self.plot_to_file(accuracy_title, loss_title, fig_name)

                # Load model and history
                self.load_dropout_history_from_file(kernel, stride)

                # Load history data
                self.load_history_data()

                # Set names for plot
                dropout_fig_name = 'cifar10_dropout_ker' + str(kernel) + '_str' + str(stride)
                dropout_loss_title = 'Loss for kernel: ' + str(kernel) + ' stride: ' + str(stride) + ' dropout'
                dropout_accuracy_title = 'Accuracy for kernel: ' + str(kernel) + ' stride: ' + str(stride) + ' dropout'

                # Plot
                self.plot_to_file(dropout_accuracy_title, dropout_loss_title, dropout_fig_name)

    def load_normal_history_from_file(self, kernel, stride):
        # Load trained data:
        log = 'cifar10_ker' + str(kernel) + '_str' + str(stride) + '.log'
        # model = 'cifar10_ker' + str(kernel) + '_str' + str(stride) + '.h5py'

        # self.model = load_model(model)
        self.training_history = pandas.read_csv(log, sep=',', engine='python')

    def load_dropout_history_from_file(self, kernel, stride):
        # Load trained data:
        dropout_log = 'cifar10_dropout_ker' + str(kernel) + '_str' + str(stride) + '.log'
        # dropout_model = 'cifar10_dropout_ker' + str(kernel) + '_str' + str(stride) + '.h5py'

        # self.model = load_model(dropout_model)
        self.training_history = pandas.read_csv(dropout_log, sep=',', engine='python')

    def load_history_data(self):
        self.accuracy = self.training_history['acc']
        self.val_accuracy = self.training_history['val_acc']
        self.loss = self.training_history['loss']
        self.val_loss = self.training_history['val_loss']
        self.epochs_plot = range(len(self.accuracy))

    def plot_to_file(self, accuracy_title, loss_title, fig_name):
        plt.figure()
        plt.plot(self.epochs_plot, self.accuracy, 'bo', label='Training accuracy')
        plt.plot(self.epochs_plot, self.val_accuracy, 'b', label='Validation accuracy')
        plt.title(accuracy_title)
        plt.legend()
        plt.savefig(fig_name + '_acc.png')
        plt.close()

        plt.figure()
        plt.plot(self.epochs_plot, self.loss, 'bo', label='Training loss')
        plt.plot(self.epochs_plot, self.val_loss, 'b', label='Validation loss')
        plt.title(loss_title)
        plt.legend()
        plt.savefig(fig_name + '_los.png')
        plt.close()


if __name__ == "__main__":
    pappa_sa_10 = Cifar10CNN()
    pappa_sa_10.train_best()
