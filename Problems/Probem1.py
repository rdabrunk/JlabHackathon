import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping

if __name__ == '__main__':

    # Read in the training and testing CSV files
    train_data = pd.read_csv('Data/P1/train.csv', header=None)
    test_data = pd.read_csv('Data/P1/test.csv', header=None)

    # Extract the hit values and reshape into a 3D tensor
    x_train = np.array(train_data.iloc[:, :900])
    x_train = np.reshape(x_train, (len(train_data), 30, 30, 1))

    x_test = np.array(test_data.iloc[:, :900])
    x_test = np.reshape(x_test, (len(test_data), 30, 30, 1))

    # Extract the target variables
    y_train = np.array(train_data.iloc[:, 900:])
    y_test = np.array(test_data.iloc[:, 900:])

    # Define the model architecture
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(30, 30, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(3, activation='linear'))

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='rmsprop')

    # Train the model
    # model.fit(x_train, y_train, epochs=10, batch_size=32)

    # Define the early stopping callback
    early_stop = EarlyStopping(monitor='val_loss', patience=15, verbose=1)

    # Train the model with the early stopping callback
    history = model.fit(x_train, y_train, epochs=1000, batch_size=64, validation_data=(x_test, y_test),
                        callbacks=[early_stop])

    # plot validation from one epoch to the next
    val_loss = history.history["val_loss"]
    loss = history.history["loss"]
    epochs = range(1, len(val_loss) + 1)
    plt.plot(epochs, val_loss, "-o", label="Validation Loss")

    # plot the loss as well
    plt.plot(epochs, loss, "-o", label="Loss")
    plt.title("Validation Accuracy and Loss")

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy and Loss")
    plt.legend()

    # save the plot to a file
    plt.savefig("val_loss.png")

    # Evaluate the model on the testing set
    score = model.evaluate(x_test, y_test, verbose=2)
    print('Test loss:', score)
