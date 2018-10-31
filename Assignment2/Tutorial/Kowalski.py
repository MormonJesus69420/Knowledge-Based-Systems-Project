import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import fashion_mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


class Kowalski:
    def analyze(self):
        (train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()

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
        # plt.title("Ground Truth : {}".format(train_Y[0]))
        #
        # # Display the first image in testing data
        # plt.subplot(122)
        # plt.imshow(test_X[0, :, :], cmap='gray')
        # plt.title("Ground Truth : {}".format(test_Y[0]))
        # plt.show()

        # Reshape data set into 28x28x1 matrices
        train_X = train_X.reshape(-1, 28, 28, 1)
        test_X = test_X.reshape(-1, 28, 28, 1)
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
        epochs = 20  # 20 originally
        num_classes = 10

        # #
        # # Without dropout:
        # # Test loss :  0.2559400901377201
        # # Test accuracy :  0.9131
        # #
        # # Create a training model layers
        # fashion_model = Sequential()
        # fashion_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=(28, 28, 1), padding='same'))
        # fashion_model.add(LeakyReLU(alpha=0.1))
        # fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        # fashion_model.add(Conv2D(64, kernel_size=(3, 3), activation='linear', padding='same'))
        # fashion_model.add(LeakyReLU(alpha=0.1))
        # fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        # fashion_model.add(Conv2D(128, kernel_size=(3, 3), activation='linear', padding='same'))
        # fashion_model.add(LeakyReLU(alpha=0.1))
        # fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        # fashion_model.add(Flatten())
        # fashion_model.add(Dense(128, activation='linear'))
        # fashion_model.add(LeakyReLU(alpha=0.1))
        # fashion_model.add(Dense(num_classes, activation='softmax'))
        #
        # # Compile model
        # fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
        #                       metrics=['accuracy'])
        #
        # # Show compiled model
        # fashion_model.summary()
        #
        # # Train the model
        # fashion_train = fashion_model.fit(train_X, train_label, batch_size=batch_size, verbose=2, epochs=epochs,
        #                                   validation_data=(valid_X, valid_label))
        # # Save the model to file
        # fashion_model.save("fashion_model.h5py")
        #
        # # Evaluate the model
        # test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=1)
        # print('Test loss : ', test_eval[0])
        # print('Test accuracy : ', test_eval[1])

        # Create a training model layers
        fashion_model = Sequential()
        fashion_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same', input_shape=(28, 28, 1)))
        fashion_model.add(LeakyReLU(alpha=0.1))
        fashion_model.add(MaxPooling2D((2, 2), padding='same'))
        fashion_model.add(Dropout(0.25))
        fashion_model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
        fashion_model.add(LeakyReLU(alpha=0.1))
        fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        fashion_model.add(Dropout(0.25))
        fashion_model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
        fashion_model.add(LeakyReLU(alpha=0.1))
        fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        fashion_model.add(Dropout(0.4))
        fashion_model.add(Flatten())
        fashion_model.add(Dense(128, activation='linear'))
        fashion_model.add(LeakyReLU(alpha=0.1))
        fashion_model.add(Dropout(0.3))
        fashion_model.add(Dense(num_classes, activation='softmax'))

        # Compile model
        fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                              metrics=['accuracy'])

        # Show compiled model
        fashion_model.summary()

        # Train the model
        fashion_train = fashion_model.fit(train_X, train_label, batch_size=batch_size, epochs=epochs, verbose=1,
                                          validation_data=(valid_X, valid_label))

        # Save the model to file
        fashion_model.save("fashion_model_dropout.h5py")

        # Evaluate the model
        test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=1)
        print('Test loss : ', test_eval[0])
        print('Test accuracy : ', test_eval[1])

        # Plot Accuracy and Loss between training and validation data.
        accuracy = fashion_train.history['acc']
        val_accuracy = fashion_train.history['val_acc']
        loss = fashion_train.history['loss']
        val_loss = fashion_train.history['val_loss']
        epochs = range(len(accuracy))
        plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
        plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()

        # Get predicted classes
        predicted_classes = fashion_model.predict(test_X)

        # Convert to integers and then choose the maximum
        predicted_classes = np.argmax(np.round(predicted_classes), axis=1)

        # Find number of correct predictions
        correct = np.where(predicted_classes == test_Y)[0]
        print(f"Found {len(correct)} correct labels")

        # Plot pictures of 9 correct predictions
        for i, correct in enumerate(correct[:9]):
            plt.subplot(3, 3, i + 1)
            plt.imshow(test_X[correct].reshape(28, 28), cmap='gray', interpolation='none')
            plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_Y[correct]))
            plt.tight_layout()

        plt.show()

        # Find number of incorrect predictions
        incorrect = np.where(predicted_classes != test_Y)[0]
        print(f"Found {len(incorrect)} incorrect labels")

        # Plot pictures of 9 incorrect predictions
        for i, incorrect in enumerate(incorrect[:9]):
            plt.subplot(3, 3, i + 1)
            plt.imshow(test_X[incorrect].reshape(28, 28), cmap='gray', interpolation='none')
            plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_Y[incorrect]))
            plt.tight_layout()

        plt.show()

        # Calculate and print classification report
        target_names = ["Class {}".format(i) for i in range(num_classes)]
        print(classification_report(test_Y, predicted_classes, target_names=target_names))


if __name__ == "__main__":
    kowalski = Kowalski()
    kowalski.analyze()
