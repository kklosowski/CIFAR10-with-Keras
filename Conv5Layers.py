from pathlib import Path

from scipy import ndimage
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def augment_dataset(train_images, train_labels):
    train_images = np.concatenate((train_images, [np.flip(x, axis=1) for x in train_images]))
    train_images = np.concatenate((train_images, np.random.normal(train_images, 0.03)))
    rot15 = [ndimage.interpolation.rotate(x, 15, reshape=False, mode="nearest") for x in train_images]
    rot345 = [ndimage.interpolation.rotate(x, 345, reshape=False, mode="nearest") for x in train_images]
    train_images = np.concatenate((train_images, rot15))
    train_images = np.concatenate((train_images, rot345))

    train_labels_2x = np.append(train_labels, train_labels)
    train_labels_4x = np.append(train_labels_2x, train_labels_2x)
    train_labels_8x = np.append(train_labels_4x, train_labels_4x)
    train_labels_12x = np.append(train_labels_8x, train_labels_4x)

    # for x in range(12):
    #     plt.imshow(train_images[x * 10000 + 57])
    #     plt.show()

    return train_images, train_labels_12x


def learning_rate_scheduler(epoch):
    if epoch < 10:
        return 0.001
    if 10 <= epoch < 20:
        return 0.0001
    else:
        return 0.00001


def main():
    # Hyper-parameters
    num_classes = 10
    epochs = 25
    learning_rate = 0.001
    batch_size = 500
    model_name = "cifar10_fc_5layer"
    data_augmentation = False

    # Dictionary of common parameters used in convolutional layers
    params_conv2d = {
        "padding": "SAME",
        "activation": keras.activations.elu
        # 'kernel_regularizer': keras.regularizers.l2(0.01)
    }

    # Data Loading
    (train_images, train_labels) = np.load("data/trnImage.npy"), np.load("data/trnLabel.npy")
    (test_images, test_labels) = np.load("data/tstImage.npy"), np.load("data/tstLabel.npy")

    # Pre-processing step
    (train_images, train_labels) = np.moveaxis(train_images, -1, 0), np.subtract(train_labels.flatten(), 1)
    (test_images, test_labels) = np.moveaxis(test_images, -1, 0), np.subtract(test_labels.flatten(), 1)

    # Dataset augmentation from 10k to 120k images
    if data_augmentation:
        train_images, train_labels = augment_dataset(train_images, train_labels)

    # Conversion of labels to one-hot arrays
    train_labels = keras.utils.to_categorical(train_labels, num_classes)
    test_labels = keras.utils.to_categorical(test_labels, num_classes)

    model = keras.Sequential([
        keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, input_shape=(32, 32, 3)),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=64, kernel_size=3, strides=2),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=128, kernel_size=3, strides=2),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=256, kernel_size=3, strides=2),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=512, kernel_size=1, strides=1),
        keras.layers.Flatten(),
        keras.layers.Dense(1000, activation=keras.activations.relu),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(500, activation=keras.activations.relu),
        keras.layers.Dense(10, activation=None),
        keras.layers.Softmax()
    ])

    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    # Loading model weights if available
    if Path("models/" + model_name + '.h5').is_file():
        print("loading")
        # model.load_weights("models/cifar10_4block_10k.e07-acc0.696.h5")
        model.load_weights("models/" + model_name + '.h5')

    # Training callbacks
    callbacks = [keras.callbacks.LearningRateScheduler(learning_rate_scheduler, verbose=0),
                 keras.callbacks.ModelCheckpoint('checkpoints/' + model_name + '.e{epoch:02d}-acc{val_categorical_accuracy:.3f}.h5',
                                                 monitor='val_categorical_accuracy',
                                                 verbose=0,
                                                 save_best_only=True,
                                                 save_weights_only=False,
                                                 mode='max',
                                                 period=epochs//20)
                 ]

    callbacks = None

    # Training step
    history = model.fit(train_images, train_labels,
                        batch_size=batch_size,
                        epochs=epochs,
                        shuffle=True,
                        validation_data=(test_images, test_labels),
                        verbose=2,
                        callbacks=callbacks)

    model.save("models/" + model_name + '.h5')

    with open('model_architecture.json', 'w') as f:
        f.write(model.to_json())

    # Accuracy over training time
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title(model_name)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()

    plt.plot(history.history['categorical_accuracy'], label="Training Accuracy")
    plt.plot(history.history['val_categorical_accuracy'], label="Validation Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label="Training Loss")
    plt.plot(history.history['val_loss'], label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

