from pathlib import Path

from scipy import  ndimage
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2


def augment_dataset(train_images, train_labels):
    train_images = np.concatenate((train_images, [np.flip(x, axis=1) for x in train_images]))
    train_images = np.concatenate((train_images, np.random.normal(train_images, 0.03)))
    rot15 = [ndimage.interpolation.rotate(x, 15, reshape=False, mode="nearest") for x in train_images]
    # rot30 = [ndimage.interpolation.rotate(x, 30, reshape=False, mode="nearest") for x in train_images]
    # rot330 = [ndimage.interpolation.rotate(x, 330, reshape=False, mode="nearest") for x in train_images]
    rot345 = [ndimage.interpolation.rotate(x, 345, reshape=False, mode="nearest") for x in train_images]
    train_images = np.concatenate((train_images, rot15))
    train_images = np.concatenate((train_images, rot30))
    train_images = np.concatenate((train_images, rot330))
    train_images = np.concatenate((train_images, rot345))

    train_labels_2x = np.append(train_labels, train_labels)
    train_labels_4x = np.append(train_labels_2x, train_labels_2x)
    train_labels_8x = np.append(train_labels_4x, train_labels_4x)
    train_labels_12x = np.append(train_labels_8x, train_labels_4x)
    # train_labels_16x = np.append(train_labels_12x, train_labels_4x)
    # train_labels_20x = np.append(train_labels_16x, train_labels_4x)

    return train_images, train_labels_12x


def main():
    num_classes = 10

    (train_images, train_labels) = np.load("data/trnImage.npy"), np.load("data/trnLabel.npy")
    (test_images, test_labels) = np.load("data/tstImage.npy"), np.load("data/tstLabel.npy")

    train_images = np.moveaxis(train_images, -1, 0)
    test_images = np.moveaxis(test_images, -1, 0)

    train_labels = train_labels.flatten()
    test_labels = test_labels.flatten()

    train_labels = np.subtract(train_labels, 1)
    test_labels = np.subtract(test_labels, 1)

    train_images, train_labels = augment_dataset(train_images, train_labels)



    print(train_images.shape, train_labels.shape)

    train_labels = keras.utils.to_categorical(train_labels, num_classes)
    test_labels = keras.utils.to_categorical(test_labels, num_classes)

    epochs = 55

    # model = keras.Sequential([
    #     keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, input_shape=(32, 32, 3)),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Conv2D(filters=64, kernel_size=3, strides=2),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Conv2D(filters=128, kernel_size=3, strides=2),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Conv2D(filters=256, kernel_size=3, strides=2),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Conv2D(filters=512, kernel_size=1, strides=1),
    #     keras.layers.Flatten(),
    #     keras.layers.Dense(1000, activation=tf.nn.leaky_relu),
    #     keras.layers.Dropout(0.5),
    #     keras.layers.Dense(500, activation=tf.nn.leaky_relu),
    #     keras.layers.Dense(10, activation=None),
    #     keras.layers.Softmax()
    #
    # ])

    x = keras.Input((32, 32, 3))
    # conv_1
    conv1 = keras.layers.Conv2D(filters=32, kernel_size=7, strides=1, padding="SAME", activation=keras.activations.relu, kernel_regularizer=keras.regularizers.l2(0.01))(x)
    # conv_2_x
    max_pool = keras.layers.MaxPool2D(padding="SAME")(conv1)
    conv2_1 = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="SAME", activation=keras.activations.relu, kernel_regularizer=keras.regularizers.l2(0.01))(max_pool)
    conv2_2 = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="SAME", activation=keras.activations.relu, kernel_regularizer=keras.regularizers.l2(0.01))(conv2_1)
    skip2_1 = keras.layers.add([max_pool, conv2_2])

    conv2_3 = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="SAME", activation=keras.activations.relu, kernel_regularizer=keras.regularizers.l2(0.01))(conv2_2)
    conv2_4 = keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding="SAME", activation=keras.activations.relu, kernel_regularizer=keras.regularizers.l2(0.01))(conv2_3)

    resize2 = keras.layers.Conv2D(filters=32, kernel_size=1, strides=2, padding="SAME")(skip2_1)
    skip2_2 = keras.layers.add([resize2, conv2_4])

    # conv_3_x
    conv3_1 = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="SAME", activation=keras.activations.relu, kernel_regularizer=keras.regularizers.l2(0.01))(skip2_2)
    conv3_2 = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="SAME", activation=keras.activations.relu, kernel_regularizer=keras.regularizers.l2(0.01))(conv3_1)

    resize3_1 = keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding="SAME")(skip2_2)
    skip3_1 = keras.layers.add([resize3_1, conv3_2])

    conv3_3 = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="SAME", activation=keras.activations.relu, kernel_regularizer=keras.regularizers.l2(0.01))(skip3_1)
    conv3_4 = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="SAME", activation=keras.activations.relu, kernel_regularizer=keras.regularizers.l2(0.01))(conv3_3)

    resize3_2 = keras.layers.Conv2D(filters=64, kernel_size=1, strides=2, padding="SAME")(skip3_1)
    skip3_2 = keras.layers.add([resize3_2, conv3_4])

    # conv_4_x
    conv4_1 = keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="SAME", activation=keras.activations.relu, kernel_regularizer=keras.regularizers.l2(0.01))(skip3_2)
    conv4_2 = keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="SAME", activation=keras.activations.relu, kernel_regularizer=keras.regularizers.l2(0.01))(conv4_1)

    resize4_1 = keras.layers.Conv2D(filters=128, kernel_size=1, strides=1, padding="SAME")(skip3_2)
    skip4_1 = keras.layers.add([resize4_1, conv4_2])

    conv4_3 = keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="SAME", activation=keras.activations.relu, kernel_regularizer=keras.regularizers.l2(0.01))(skip4_1)
    conv4_4 = keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding="SAME", activation=keras.activations.relu, kernel_regularizer=keras.regularizers.l2(0.01))(conv4_3)

    resize4_2 = keras.layers.Conv2D(filters=128, kernel_size=1, strides=2, padding="SAME")(skip4_1)
    skip4_2 = keras.layers.add([resize4_2, conv4_4])

    # # conv_5_x
    # conv5_1 = keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="SAME", activation=keras.activations.relu)(skip4_2)
    # conv5_2 = keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="SAME")(conv5_1)
    #
    # resize5_1 = keras.layers.Conv2D(filters=256, kernel_size=1, strides=1, padding="SAME")(skip4_2)
    # skip5_1 = keras.layers.add([resize5_1, conv5_2])
    #
    # conv5_3 = keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="SAME", activation=keras.activations.relu)(skip5_1)
    # conv5_4 = keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, padding="SAME")(conv5_3)
    #
    # resize5_2 = keras.layers.Conv2D(filters=256, kernel_size=1, strides=2, padding="SAME")(skip5_1)
    # skip5_2 = keras.layers.add([resize5_2, conv5_4])

    avg_pool = keras.layers.AvgPool2D(strides=2)(skip3_2)
    flat = keras.layers.Flatten()(avg_pool)
    dense256 = keras.layers.Dense(256)(flat)
    dense10 = keras.layers.Dense(10)(dense256)
    softmax = keras.layers.Softmax()(dense10)

    model = keras.Model(inputs=x, outputs=softmax)

    # if Path("models/cifar10.h5").is_file():
    #     print("loading")
    #     model.load_weights("models/cifar10_70percent.h5")

    model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    history = model.fit(train_images, train_labels, batch_size=1000, epochs=epochs, shuffle=True, validation_data=(test_images, test_labels), verbose=2)
    model.save("models/cifar10_70percent200k.h5")

    plt.plot(history.history['val_categorical_accuracy'])
    plt.show()


if __name__ == '__main__':
    main()
