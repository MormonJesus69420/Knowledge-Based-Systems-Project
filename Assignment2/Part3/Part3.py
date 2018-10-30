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
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

c10 = cifar10.load_data()
train_X, train_Y, test_X, test_Y = (b for a in c10 for b in a)

# Check training and testing data shape
print('Training data shape : ', train_X.shape, train_Y.shape)
print('Testing data shape : ', test_X.shape, test_Y.shape)

# Find the number of unique labels from the training label set
classes = np.unique(train_Y)
print('Total number of outputs : ', len(classes))
print('Output classes : ', classes)

# # Setup pyplot for displaying images
# plt.figure(figsize=[5, 5])
#
# # Display the first image in training data
# plt.subplot(121)
# plt.imshow(train_X[0, :, :], cmap='gray')
# plt.title("Train Pic : {}".format(train_Y[0]))
#
# # Display the first image in testing data
# plt.subplot(122)
# plt.imshow(test_X[0, :, :], cmap='gray')
# plt.title("Test Pic : {}".format(test_Y[0]))
# plt.show()

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

#for kernel in [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]:
for kernel in [(6, 6), (7, 7), (8, 8)]:
    for stride in [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]:
        log = 'cifar10_ker' + str(kernel) + '_str' + str(stride) + '.log'
        model = 'cifar10_ker' + str(kernel) + '_str' + str(stride) + '.h5py'
        loss_title = 'Loss for kernel: ' + str(kernel) + ' stride: ' + str(stride)
        accuracy_title = 'Accuracy for kernel: ' + str(kernel) + ' stride: ' + str(stride)
        dropout_log = 'cifar10_dropout_ker' + str(kernel) + '_str' + str(stride) + '.log'
        dropout_model = 'cifar10_dropout_ker' + str(kernel) + '_str' + str(stride) + '.h5py'
        dropout_loss_title = 'Loss for kernel: ' + str(kernel) + ' stride: ' + str(stride) + ' dropout'
        dropout_accuracy_title = 'Accuracy for kernel: ' + str(kernel) + ' stride: ' + str(stride) + ' dropout'
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

        # Plot Accuracy and Loss between training and validation data.
        accuracy = training_history.history['acc']
        val_accuracy = training_history.history['val_acc']
        loss = training_history.history['loss']
        val_loss = training_history.history['val_loss']
        epochs_plot = range(len(accuracy))
        plt.plot(epochs_plot, accuracy, 'bo', label='Training accuracy')
        plt.plot(epochs_plot, val_accuracy, 'b', label='Validation accuracy')
        plt.title(accuracy_title)
        plt.legend()
        plt.figure()
        plt.plot(epochs_plot, loss, 'bo', label='Training loss')
        plt.plot(epochs_plot, val_loss, 'b', label='Validation loss')
        plt.title(loss_title)
        plt.legend()
        plt.show()

        # Evaluate the model
        test_eval = cifar10_model.evaluate(test_X, test_Y_one_hot, verbose=1)
        print('Test loss kernel: ' + str(kernel) + ' stride: ' + str(stride) + ': ', test_eval[0])
        print('Test accuracy kernel: ' + str(kernel) + ' stride: ' + str(stride) + ': ', test_eval[1])

        # # Get predicted classes
        # predicted_classes = cifar10_model.predict(test_X)
        #
        # # Calculate and print classification report
        # target_names = ["Class {}".format(i) for i in range(num_classes)]
        # print(classification_report(test_Y, predicted_classes, target_names=target_names))

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # Train with dropout:
        # Create a training model layers
        cifar10_model = Sequential()
        cifar10_model.add(
                Conv2D(32, kernel_size=kernel, strides=stride, activation='linear', padding='same',
                       input_shape=(32, 32, 3)))
        cifar10_model.add(LeakyReLU(alpha=0.1))
        cifar10_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        cifar10_model.add(Dropout(0.25))
        cifar10_model.add(Conv2D(64, kernel_size=kernel, strides=stride, activation='linear', padding='same'))
        cifar10_model.add(LeakyReLU(alpha=0.1))
        cifar10_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        cifar10_model.add(Dropout(0.25))
        cifar10_model.add(Conv2D(128, kernel_size=kernel, strides=stride, activation='linear', padding='same'))
        cifar10_model.add(LeakyReLU(alpha=0.1))
        cifar10_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        cifar10_model.add(Dropout(0.25))
        cifar10_model.add(Flatten())
        cifar10_model.add(Dense(128, activation='linear'))
        cifar10_model.add(LeakyReLU(alpha=0.1))
        cifar10_model.add(Dropout(0.1))
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

        # Plot Accuracy and Loss between training and validation data.
        accuracy = training_history.history['acc']
        val_accuracy = training_history.history['val_acc']
        loss = training_history.history['loss']
        val_loss = training_history.history['val_loss']
        epochs_plot = range(len(accuracy))
        plt.plot(epochs_plot, accuracy, 'bo', label='Training accuracy')
        plt.plot(epochs_plot, val_accuracy, 'b', label='Validation accuracy')
        plt.title(dropout_accuracy_title)
        plt.legend()
        plt.figure()
        plt.plot(epochs_plot, loss, 'bo', label='Training loss')
        plt.plot(epochs_plot, val_loss, 'b', label='Validation loss')
        plt.title(dropout_loss_title)
        plt.legend()
        plt.show()

        # Evaluate the model
        test_eval = cifar10_model.evaluate(test_X, test_Y_one_hot, verbose=1)
        print('Test loss kernel: ' + str(kernel) + ' stride: ' + str(stride) + ' dropout: ', test_eval[0])
        print('Test accuracy kernel: ' + str(kernel) + ' stride: ' + str(stride) + ' dropout: ', test_eval[1])

        # # Get predicted classes
        # predicted_classes = cifar10_model.predict(test_X)
        #
        # # Calculate and print classification report
        # target_names = ["Class {}".format(i) for i in range(num_classes)]
        # print(classification_report(test_Y, predicted_classes, target_names=target_names))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # Load trained data:
# cifar10_model = load_model("cifar10_dropout_model.h5py")
# training_history = pandas.read_csv('training.log', sep=',', engine='python')
# accuracy = training_history['acc']
# val_accuracy = training_history['val_acc']
# loss = training_history['loss']
# val_loss = training_history['val_loss']
