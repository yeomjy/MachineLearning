import glob

import numpy as np


def get_idx(n):
    np.random.seed(0)
    indices = list(range(n))
    np.random.shuffle(indices)
    train_size = int(0.6 * n)
    test_size = int(0.2 * n)
    train_idx = indices[0: train_size]
    test_idx = indices[train_size: train_size + test_size]
    val_idx = indices[train_size + test_size: n]
    return train_idx, test_idx, val_idx


if __name__ == '__main__':
    image_path = glob.glob(r'./unet_data/image/*/*.jpg')
    mask_path = glob.glob(r'./unet_data/masks/*/*.png')
    n_img = len(image_path)
    n_mask = len(mask_path)
    if n_img != n_mask:
        raise ValueError("Invalid Image or mask")
    train_idx, test_idx, val_idx = get_idx(n_img)

    img_f = open('./unet_data/train_image_list', 'w')
    mask_f = open('./unet_data/train_mask_list', 'w')
    for i in train_idx:
        img_f.write(image_path[i] + '\n')
        mask_f.write(mask_path[i] + '\n')

    img_f.close()
    mask_f.close()

    img_f = open('./unet_data/test_image_list', 'w')
    mask_f = open('./unet_data/test_mask_list', 'w')
    for i in test_idx:
        img_f.write(image_path[i] + '\n')
        mask_f.write(mask_path[i] + '\n')

    img_f.close()
    mask_f.close()

    img_f = open('./unet_data/val_image_list', 'w')
    mask_f = open('./unet_data/val_mask_list', 'w')
    for i in val_idx:
        img_f.write(image_path[i] + '\n')
        mask_f.write(mask_path[i] + '\n')

    img_f.close()
    mask_f.close()

    # check
    total_img = 0
    total_mask = 0

    img_f = open('./unet_data/train_image_list', 'r')
    mask_f = open('./unet_data/train_mask_list', 'r')
    img_l = img_f.readlines()
    mask_l = mask_f.readlines()
    total_img += len(img_l)
    total_mask += len(mask_l)
    print(f'Train Dataset: {len(img_l)} images and {len(mask_l)} masks')
    img_f.close()
    mask_f.close()

    img_f = open('./unet_data/test_image_list', 'r')
    mask_f = open('./unet_data/test_mask_list', 'r')
    img_l = img_f.readlines()
    mask_l = mask_f.readlines()
    total_img += len(img_l)
    total_mask += len(mask_l)
    print(f'Test Dataset: {len(img_l)} images and {len(mask_l)} masks')
    img_f.close()
    mask_f.close()

    img_f = open('./unet_data/val_image_list', 'r')
    mask_f = open('./unet_data/val_mask_list', 'r')
    img_l = img_f.readlines()
    mask_l = mask_f.readlines()
    total_img += len(img_l)
    total_mask += len(mask_l)
    print(f'Validation Dataset: {len(img_l)} images and {len(mask_l)} masks')
    img_f.close()
    mask_f.close()

    print(f'Total Length: {total_img} images and {total_mask} masks')
    print(f'Orig: {n_img} images and {n_mask} masks')
