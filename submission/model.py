import csv
import numpy as np
import math
import cv2
import keras
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

DEBUG = True


def load_data():
    """
    Load data from specified folders. Balance the driving data by discarding some of the data on straight road.
    Use the images from left/right cameras to learn about recovery behavior.
    """
    data_folders = ['./udacity_data/']

    # Extract records from csv file
    # Each records contains [center, left, right, measurement, throttle, brake, speed]
    image_files, measurements = [], []
    for data_folder in data_folders:
        with open(data_folder + 'driving_log.csv') as f:
            reader = csv.reader(f)
            next(reader)  # skip header row
            for row in reader:
                # Get the measurement
                measurement = float(row[3])
                if abs(measurement) < 1e-3:
                    prob = np.random.uniform(0, 1)
                    if prob > 0.15:
                        continue
                # For images from center camera
                image_files.append(row[0])
                measurements.append(measurement)
                # For images from left camera
                image_files.append(row[1])
                measurements.append(measurement + 0.2)
                # For images from right camera
                image_files.append(row[2])
                measurements.append(measurement - 0.2)

    # Split train/validation set
    return train_test_split(image_files, measurements, test_size=0.1)


def preprocessing(image):
    """
    Pre-process the image with cropping, resizing, and converting to YUV color space.
    """
    # Crop the image
    image = image[70:135, :, :]

    # Resize the image
    image = cv2.resize(image, (200, 66))

    # Convert image to YUV color space
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

    return image


def distort_image(image, measurement):
    """
    Distort the image and adjust the corresponding steering angle to provide extra data.
    """
    # Add random brightness
    # brightness = np.random.randint(-50, 50)
    # y_ch = image[:, :, 0].astype(np.int16)
    # y_ch += brightness
    # y_ch[y_ch > 255] = 255
    # y_ch[y_ch < 0] = 0
    # image[:, :, 0] = y_ch.astype(np.uint8)

    # Randomly flip the image
    # if np.random.uniform(0, 1) > 0.5:
    #     image = cv2.flip(image, 1)
    #     measurement = -1.0 * measurement

    # Shift the image left/right
    offset = np.random.randint(-20, 20)
    measurement += offset * 0.005

    return image, measurement


def generator(image_files, measurements, batch_size=128, training=False):
    """
    Training/testing data generator.
    """
    n_samples = len(image_files)
    while True:
        image_files, measurements = shuffle(image_files, measurements)

        for offset in range(0, n_samples, batch_size):
            image_files_batch = image_files[offset:offset + batch_size]
            measurements_batch = measurements[offset:offset + batch_size]

            X, y = [], []
            for image_file, measurement in zip(image_files_batch, measurements_batch):
                image = mpimg.imread(image_file)
                image = preprocessing(image)
                if training:
                    image, measurement = distort_image(image, measurement)
                X.append(image)
                y.append(measurement)
            X = np.array(X)
            y = np.array(y)
            yield shuffle(X, y)


def nvidia_end_to_end(input_shape):
    """
    Build the model for Nvidia End-to-End network from
    Bojarski, Mariusz, et al. "End to end learning for self-driving cars." arXiv preprint arXiv:1604.07316 (2016).
    """
    model = Sequential()
    model.add(Conv2D(24,
                     kernel_size=(5, 5),
                     strides=(2, 2),
                     padding='valid',
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(36,
                     kernel_size=(5, 5),
                     strides=(2, 2),
                     padding='valid',
                     activation='relu'))
    model.add(Conv2D(48,
                     kernel_size=(5, 5),
                     strides=(2, 2),
                     padding='valid',
                     activation='relu'))
    model.add(Conv2D(64,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='valid',
                     activation='relu'))
    model.add(Conv2D(64,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='valid',
                     activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adam(lr=0.001))
    return model


def visualize_data(image_files, measurements, folder):
    # Plot data distribution
    hist, edges = np.histogram(measurements, bins=50)
    width = 0.7 * (edges[1] - edges[0])
    center = (edges[:-1] + edges[1:]) / 2
    fig = plt.figure()
    plt.bar(center, hist, align='center', width=width)
    plt.title('Training data distribution')
    plt.savefig(folder + 'histogram.jpg', bbox_inches='tight')

    # Show some training images with measurement
    indices = np.random.choice(len(image_files), size=9, replace=False)
    _image_files = [image_files[i] for i in indices]
    _measurements = [measurements[i] for i in indices]
    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(hspace=0.0)
    for i in range(9):
        image = mpimg.imread(_image_files[i])
        ax = plt.subplot(3, 3, i + 1)
        ax.imshow(image)
        ax.set_xlabel('steering={:.2f}'.format(_measurements[i]))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(folder + 'sample_images.jpg', bbox_inches='tight')

    # Plot the same training images with measurement after preprocessing
    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(hspace=0.0)
    for i in range(9):
        image = mpimg.imread(_image_files[i])
        measurement = _measurements[i]
        image = preprocessing(image)
        ax = plt.subplot(3, 3, i + 1)
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_YUV2RGB))
        ax.set_xlabel('steering={:.2f}'.format(measurement))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(folder + 'sample_images_preprocessed.jpg', bbox_inches='tight')

    # Plot the same training images with measurement after distortion
    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(hspace=0.0)
    for i in range(9):
        image = mpimg.imread(_image_files[i])
        measurement = _measurements[i]
        image = preprocessing(image)
        image, measurement = distort_image(image, measurement)
        ax = plt.subplot(3, 3, i + 1)
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_YUV2RGB))
        ax.set_xlabel('steering={:.2f}'.format(measurement))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(folder + 'sample_images_distorted.jpg', bbox_inches='tight')


def main():
    # Get training and validation data set
    train_image_files, validation_image_files, train_measurements, validation_measurements = load_data()
    print('Training set has {} samples'.format(len(train_image_files)))
    print('Validation set has {} samples'.format(len(validation_image_files)))

    submission_folder = './submission/'
    # Dataset visualization
    if DEBUG:
        visualize_data(train_image_files,
                       train_measurements, submission_folder)

    # Hyperprameters
    batch_size = 128
    epochs = 10

    # Generators
    train_generator = generator(
        train_image_files, train_measurements, batch_size=batch_size, training=True)
    validation_generator = generator(
        validation_image_files, validation_measurements, batch_size=batch_size, training=False)

    # Print the keras version
    print('keras version: {}'.format(keras.__version__))

    # Image properties
    image_height = 66
    image_width = 200
    image_channels = 3
    input_shape = (image_height, image_width, image_channels)

    # Build the network
    model = nvidia_end_to_end(input_shape)

    steps_per_epoch = len(train_image_files) // batch_size
    validation_steps = len(validation_image_files) // batch_size
    history_object = model.fit_generator(train_generator,
                                         steps_per_epoch=steps_per_epoch,
                                         epochs=epochs,
                                         verbose=1,
                                         validation_data=validation_generator,
                                         validation_steps=validation_steps,
                                         workers=1,
                                         use_multiprocessing=False)

    # Save the model
    model.save('model.h5')

    # Plot the training and validation loss for each epoch
    plt.figure()
    plt.plot(np.log(history_object.history['loss']))
    plt.plot(np.log(history_object.history['val_loss']))
    plt.title('Mean squared error')
    plt.ylabel('Log mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig(submission_folder + 'training_result.png', bbox_inches='tight')


if __name__ == '__main__':
    main()
