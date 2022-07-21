"""ctdata dataset."""

import tensorflow_datasets as tfds
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

# TODO(ctdata): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(ctdata): BibTeX citation
_CITATION = """
"""


class Ctdata(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for ctdata dataset."""

  VERSION = tfds.core.Version('1.0.1')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
      '1.0.1': 'Use CV2'
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(ctdata): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(512, 512, 1), use_colormap=True),
            'mask': tfds.features.LabeledImage(shape=(512, 512, 1), labels=['background', 'liver', 'spleen', 'kidney']),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'mask'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(ctdata): Downloads the data and defines the splits
    # path = dl_manager.download_and_extract('https://todo-data-url')
    base_dir = Path.home() / Path('data/ctdata/segmentation')
    images_dir = base_dir
    annotations_dir = base_dir

    train_image_list = base_dir / 'train_image_list.txt'
    train_mask_list = base_dir / 'train_mask_list.txt'

    test_image_list = base_dir / 'test_image_list.txt'
    test_mask_list = base_dir / 'test_mask_list.txt'

    val_image_list = base_dir / 'val_image_list.txt'
    val_mask_list = base_dir / 'val_mask_list.txt'


    return {
        'train': self._generate_examples(base_dir, train_image_list, train_mask_list),
        'test': self._generate_examples(base_dir, test_image_list, test_mask_list),
        'valid': self._generate_examples(base_dir, val_image_list, val_mask_list)
    }

  def _generate_examples(self, path, image_list, mask_list):
    """Yields examples."""
    f = open(image_list, 'r')

    img_l = f.readlines()
    f.close()
    f = open(mask_list, 'r')
    mask_l = f.readlines()
    f.close()
    img_l = [i.strip() for i in img_l]
    mask_l = [i.strip() for i in mask_l]

    for img, mask in zip(img_l, mask_l):
        rimg = cv2.imread(str(path / img), cv2.IMREAD_GRAYSCALE)
        rmask = cv2.imread(str(path / mask), cv2.IMREAD_GRAYSCALE)
        rmask[rmask == 38] = 1
        rmask[rmask == 75] = 2
        rmask[rmask == 113] = 3
        if len(rimg.shape) == 2:
            rimg = np.expand_dims(rimg, 2)

        if len(rmask.shape) == 2:
            rmask = np.expand_dims(rmask, 2)
        yield img, {
            'image': rimg,
            'mask': rmask
        }



    # for f in path.glob('*.jpeg'):
    #   yield 'key', {
    #       'image': f,
    #       'label': 'yes',
    #   }
