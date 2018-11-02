import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import CSVLogger
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from Loader import Loader

if __name__ == "__main__":
    loader = Loader()

    # Load data
    train_x = loader._train_x
    test_x = loader.test_x
    train_y = loader.train_y
    test_y = loader.test_y

    # Check training and testing data shape
    print('Training data shape : ', train_x.shape, train_y.shape)
    print('Testing data shape : ', test_x.shape, test_y.shape)

    # Find the unique numbers from the train labels
    classes = np.unique(train_y)
    print('Total number of outputs : ', len(classes))
    print('Output classes : ', classes)

    # Convert train_X and test_x to float32 and normalize to be between 0 and 1
    train_x = train_x.astype('float32')
    test_x = test_x.astype('float32')
    train_x = train_x / 255.0
    test_x = test_x / 255.0

    # Change the labels from categorical to one-hot encoding
    train_y_one_hot = to_categorical(train_y)
    test_y_one_hot = to_categorical(test_y)

    # Display the change for category label using one-hot encoding
    print('Original label:', train_y[0])
    print('After conversion to one-hot:', train_y_one_hot[0])

    train_x, valid_x, train_label, valid_label = train_test_split(train_x, train_y_one_hot, test_size=0.2,
                                                                  random_state=13)

    # Define some constants to be used in code
    batch_size = 512  # 64 originally
    epochs = 40  # 20 originally
    num_classes = 3

    # Create a training model layers
    model = Sequential()
    model.add(
            Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='linear', padding='same',
                   input_shape=(32, 32, 3)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='linear', padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='linear', padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    # Show compiled model
    model.summary()

    # Train the model
    csv_logger = CSVLogger('letter.log', separator=',', append=False)
    letter_train = model.fit(train_x, train_label, batch_size=batch_size, epochs=epochs, verbose=1,
                             validation_data=(valid_x, valid_label), callbacks=[csv_logger])

    # Save the model to file
    model.save('letter.h5py')

    # Evaluate the model
    test_eval = model.evaluate(test_x, test_y_one_hot, verbose=1)

    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])

    # Plot loss and accuracy of the model
    accuracy = letter_train.history['acc']
    val_accuracy = letter_train.history['val_acc']
    loss = letter_train.history['loss']
    val_loss = letter_train.history['val_loss']
    epochs_plot = range(len(accuracy))

    plt.figure()
    plt.plot(epochs_plot, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs_plot, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig('letter_acc.png')
    plt.close()

    plt.figure()
    plt.plot(epochs_plot, loss, 'bo', label='Training loss')
    plt.plot(epochs_plot, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('letter_loss.png')
    plt.close()
