from UNet import *
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

COLORMAP = [
    [0, 0, 0],
    [255, 0, 0],
    [0, 255, 0],
    [255, 255, 0]
]

def mask_to_colormap(mask):
    mask_shape = mask.shape
    colormap = np.zeros((mask_shape[0], mask_shape[1], 3))
    for i in range(mask_shape[0]):
        for j in range(mask_shape[1]):
            colormap[i][j] = COLORMAP[mask[i][j][0]]
    return colormap


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # clear_output(wait=True)
        show_predictions()
        print('\n에포크 이후 예측 예시 {}\n'.format(epoch + 1))


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask,
                 create_mask(model.predict(sample_image[tf.newaxis, ...]))])


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        if i == 0:
            plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]), cmap='gray')
        else:
            plt.imshow(tf.keras.preprocessing.image.array_to_img(mask_to_colormap(display_list[i].numpy())))
        plt.axis('off')
    plt.show()


@tf.function
def load_image_train(datapoint):
    input_image, input_mask = datapoint['image'], datapoint['mask']
    input_image = tf.image.resize(input_image, (256, 256), method='nearest')
    input_mask = tf.image.resize(input_mask, (256, 256), method='nearest')

    if np.random.random() > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image = normalize(input_image)

    return input_image, input_mask


def normalize(input_image):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image


def load_image_test(datapoint):
    input_image, input_mask = datapoint['image'], datapoint['mask']
    input_image = tf.image.resize(input_image, (256, 256), method='nearest')
    input_mask = tf.image.resize(input_mask, (256, 256), method='nearest')

    input_image = normalize(input_image)

    return input_image, input_mask


if __name__ == '__main__':
    # dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
    dataset, info = tfds.load('ctdata', with_info=True)

    TRAIN_LENGTH = info.splits['train'].num_examples
    BATCH_SIZE = 8
    BUFFER_SIZE = 1000
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
    print(info)

    train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test = dataset['test'].map(load_image_test)

    train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_dataset = test.batch(BATCH_SIZE)
    val = dataset['valid'].map(load_image_test)
    val_dataset = val.batch(BATCH_SIZE)


    # print(info.features['segmentation_mask'].num_classes)
    # print(info.features['segmentation_mask'].shape)
    # print(info.features['mask'].dtype)

    shape = (256, 256, 1)

    for image, mask in train.take(1):
        shape = image.shape
        sample_image, sample_mask = image, mask
        display([sample_image, sample_mask])

    inputs = keras.Input(shape=shape)
    outputs = unet_functional(inputs, num_classes=4)
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    EPOCHS = 10
    VAL_SUBSPLITS = 5
    VALIDATION_STEPS = info.splits['test'].num_examples // BATCH_SIZE // VAL_SUBSPLITS
    tf.keras.utils.plot_model(model, show_shapes=True)

    model_history = model.fit(train_dataset, epochs=EPOCHS,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              validation_steps=VALIDATION_STEPS,
                              validation_data=val_dataset,
                              callbacks=[DisplayCallback()])

    model.save('test')
