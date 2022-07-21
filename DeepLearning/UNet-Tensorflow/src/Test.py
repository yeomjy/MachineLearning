import tensorflow_datasets as tfds
import tensorflow as tf

from main import display, create_mask, mask_to_colormap, load_image_test
from UNet import unet_functional

def show_predictions(dataset=None, num=1, model=None):
    for image, mask in dataset.take(num):
        pred_mask = model.predict(image)
        display([image[0], mask[0], create_mask(pred_mask)])

if __name__ == '__main__':
    dataset, info = tfds.load('ctdata', with_info=True)
    test = dataset['test'].map(load_image_test)
    test_dataset = test.batch(8)
    model = tf.keras.models.load_model('test')
    model.summary()

    show_predictions(test_dataset, num=8, model=model)