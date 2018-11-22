from pathlib import Path

from scipy import ndimage
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools



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

    for x in range(12):
        plt.imshow(train_images[x * 10000 + 57])
        plt.show()

    return train_images, train_labels_12x


def learning_rate_scheduler(epoch):
    if epoch < 200:
        return 0.001
    if 200 <= epoch < 350:
        return 0.0001
    else:
        return 0.00001


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def main():
    # Hyper-parameters
    num_classes = 10
    epochs = 10
    learning_rate = 0.001
    batch_size = 500
    model_name = "cifar10_4block_10k.e35-acc0.708"
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

    # Input tensor
    x = keras.Input((32, 32, 3))
    # conv_1
    conv1 = keras.layers.Conv2D(filters=32, kernel_size=7, strides=1, **params_conv2d)(x)
    conv1 = keras.layers.SpatialDropout2D(.5, data_format='channels_last')(conv1)

    # conv_2_x
    max_pool = keras.layers.MaxPool2D(padding="SAME")(conv1)
    conv2_1 = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, **params_conv2d)(max_pool)
    conv2_1 = keras.layers.SpatialDropout2D(.5, data_format='channels_last')(conv2_1)
    conv2_2 = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, **params_conv2d)(conv2_1)
    conv2_2 = keras.layers.SpatialDropout2D(.5, data_format='channels_last')(conv2_2)
    skip2_1 = keras.layers.BatchNormalization()(keras.layers.add([max_pool, conv2_2]))

    conv2_3 = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, **params_conv2d)(conv2_2)
    conv2_3 = keras.layers.SpatialDropout2D(.5, data_format='channels_last')(conv2_3)
    conv2_4 = keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, **params_conv2d)(conv2_3)
    conv2_4 = keras.layers.SpatialDropout2D(.5, data_format='channels_last')(conv2_4)

    resize2 = keras.layers.Conv2D(filters=32, kernel_size=1, strides=2, padding="SAME")(skip2_1)
    skip2_2 = keras.layers.BatchNormalization()(keras.layers.add([resize2, conv2_4]))

    # conv_3_x
    conv3_1 = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, **params_conv2d)(skip2_2)
    conv3_1 = keras.layers.SpatialDropout2D(.5, data_format='channels_last')(conv3_1)
    conv3_2 = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, **params_conv2d)(conv3_1)
    conv3_2 = keras.layers.SpatialDropout2D(.5, data_format='channels_last')(conv3_2)

    resize3_1 = keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding="SAME")(skip2_2)
    skip3_1 = keras.layers.BatchNormalization()(keras.layers.add([resize3_1, conv3_2]))

    conv3_3 = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, **params_conv2d)(skip3_1)
    conv3_3 = keras.layers.SpatialDropout2D(.5, data_format='channels_last')(conv3_3)
    conv3_4 = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, **params_conv2d)(conv3_3)
    conv3_4 = keras.layers.SpatialDropout2D(.5, data_format='channels_last')(conv3_4)

    resize3_2 = keras.layers.Conv2D(filters=64, kernel_size=1, strides=2, padding="SAME")(skip3_1)
    skip3_2 = keras.layers.BatchNormalization()(keras.layers.add([resize3_2, conv3_4]))

    # conv_4_x
    conv4_1 = keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, **params_conv2d)(skip3_2)
    conv4_1 = keras.layers.SpatialDropout2D(.5, data_format='channels_last')(conv4_1)
    conv4_2 = keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, **params_conv2d)(conv4_1)
    conv4_2 = keras.layers.SpatialDropout2D(.5, data_format='channels_last')(conv4_2)

    resize4_1 = keras.layers.Conv2D(filters=128, kernel_size=1, strides=1, padding="SAME")(skip3_2)
    skip4_1 = keras.layers.BatchNormalization()(keras.layers.add([resize4_1, conv4_2]))

    conv4_3 = keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, **params_conv2d)(skip4_1)
    conv4_3 = keras.layers.SpatialDropout2D(.5, data_format='channels_last')(conv4_3)
    conv4_4 = keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, **params_conv2d)(conv4_3)
    conv4_4 = keras.layers.SpatialDropout2D(.5, data_format='channels_last')(conv4_4)

    resize4_2 = keras.layers.Conv2D(filters=128, kernel_size=1, strides=2, padding="SAME")(skip4_1)
    skip4_2 = keras.layers.BatchNormalization()(keras.layers.add([resize4_2, conv4_4]))

    # conv_5_x
    conv5_1 = keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, **params_conv2d)(skip4_2)
    conv5_1 = keras.layers.SpatialDropout2D(.5, data_format='channels_last')(conv5_1)
    conv5_2 = keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, **params_conv2d)(conv5_1)
    conv5_2 = keras.layers.SpatialDropout2D(.5, data_format='channels_last')(conv5_2)

    resize5_1 = keras.layers.Conv2D(filters=256, kernel_size=1, strides=1, padding="SAME")(skip4_2)
    skip5_1 = keras.layers.BatchNormalization()(keras.layers.add([resize5_1, conv5_2]))

    conv5_3 = keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, **params_conv2d)(skip5_1)
    conv5_3 = keras.layers.SpatialDropout2D(.5, data_format='channels_last')(conv5_3)
    conv5_4 = keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, **params_conv2d)(conv5_3)
    conv5_4 = keras.layers.SpatialDropout2D(.5, data_format='channels_last')(conv5_4)

    resize5_2 = keras.layers.Conv2D(filters=256, kernel_size=1, strides=1, **params_conv2d)(skip5_1)
    skip5_2 = keras.layers.BatchNormalization()(keras.layers.add([resize5_2, conv5_4]))

    avg_pool = keras.layers.AvgPool2D(strides=2)(skip4_2)
    flat = keras.layers.Flatten()(avg_pool)
    # dense256 = keras.layers.Dense(512)(flat)
    # dense10 = keras.layers.Dense(10)(dense256)
    dense10 = keras.layers.Dense(10)(flat)
    softmax = keras.layers.Softmax()(dense10)

    model = keras.Model(inputs=x, outputs=softmax)

    # Loading model weights if available
    if Path("models/" + model_name + '.h5').is_file():
        print("loading")
        # model.load_weights("models/cifar10_4block_10k.e07-acc0.696.h5")
        model.load_weights("models/" + model_name + '.h5')

    # Model compilation
    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    # Training callbacks
    callbacks = [keras.callbacks.LearningRateScheduler(learning_rate_scheduler, verbose=0),
                 # keras.callbacks.ModelCheckpoint('checkpoints/' + model_name + '.e{epoch:02d}-acc{val_categorical_accuracy:.3f}.h5',
                 #                                 monitor='val_categorical_accuracy',
                 #                                 verbose=0,
                 #                                 save_best_only=True,
                 #                                 save_weights_only=False,
                 #                                 mode='max',
                 #                                 period=epochs//20)
                 ]

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

    predictions = np.argmax(model.predict_on_batch(test_images), 1)
    conf_matrix = confusion_matrix(np.load("data/tstLabel.npy"), predictions)
    conf_matrix = np.delete(conf_matrix, 10, 1)
    conf_matrix = np.delete(conf_matrix, 0, 0)

    plot_confusion_matrix(conf_matrix, ["Plane","Car","Bird","Cat","Deer","Dog","Frog","Horse","Ship","Truck"], cmap=plt.cm.Greys)
    plt.show()


if __name__ == '__main__':
    main()

