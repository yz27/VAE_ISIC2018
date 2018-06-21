"""
    Created by yuchen on 6/20/18
    Description: This script generate the correct dataset structure for VAE outlier experiments

    souce directory structure:
    src_dir
    ├── test
    │   ├── class1
    │   │   └── 1.jpg
    │   ├── class2
    │   │   └── 1.jpg
    │   └── class3
    │       └── 1.jpg
    └── train
        ├── class1
        │   └── 1.jpg
        ├── class2
        │   └── 1.jpg
        └── class3
            └── 1.jpg

    The output folder has following structure:
    tar_dir
    ├── class1_outlier
    ├── class2_outlier
    └── class3_outlier

    where each one of them is:
    classN_outlier
    ├── vae_test
    │   ├── abnormal (10% of images of class N)
    │   └── normal (10% of images without class N)
    └── vae_train
        ├── val
        │   └── normal (10% of images without class N)
        └── train
            └── normal (80% of images without class N)

    We avoid copy images by create symlink to the images in src_dir
"""
import argparse
import os
from os.path import join as pathjoin
import glob
import random
from tqdm import tqdm
import logging
logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--src_dir', required=True, type=str)
parser.add_argument('--tar_dir', required=True, type=str)
args = parser.parse_args()
SRC_DIR = args.src_dir
TAR_DIR = args.tar_dir

# check if src_dir exists
if not os.path.isdir(SRC_DIR):
    logging.error("src_dir %s not exists!", SRC_DIR)
    exit()

# check if tar_dir exists
if os.path.exists(TAR_DIR):
    logging.error("tar_dir %s exists!", TAR_DIR)
    exit()
os.mkdir(TAR_DIR)

# list all classes
classes = os.listdir(pathjoin(SRC_DIR, 'train'))
if '.DS_Store' in classes:
    classes.remove('.DS_Store')
classes.sort()
img_paths = [os.path.abspath(path) for path in glob.iglob(pathjoin(SRC_DIR, '**', '*.jpg'), recursive=True)]

# shuffle imgs 5 times
for _ in range(5):
    random.shuffle(img_paths)

# split 90 - 10 train/test split
train_split = int(0.8 * len(img_paths))
val_split = int(0.9 * len(img_paths))
img_train_paths = img_paths[:train_split]
img_val_paths = img_paths[train_split:val_split]
img_test_paths = img_paths[val_split:]
logging.info('num. train images: %d\tnum. val images: %d\tnum. test images: %d',
             len(img_train_paths), len(img_val_paths), len(img_test_paths))

def get_class(img_path, classes):
    """
    get class name from the image path
    """
    for cls in classes:
        if cls in img_path:
            return cls
    logging.error("img_path %s is invalid!", img_path)

def symlink_without_replace(src_file, tar_dir):
    """
    create a symlink in `tar_dir` as `tar_dir/src_file`. If there exists
    `src_file` in `tar_dir`, use `tar_dir/1_src_file`, `tar_dir/11_src_file` etc.
    :param src_file: a path to a file
    :param tar_dir: a path to a dir
    """
    assert os.path.isfile(src_file)
    assert os.path.isdir(tar_dir)
    img_id = img_paths.index(src_file)
    file_name = '{}.jpg'.format(img_id)
    if file_name in os.listdir(tar_dir):
        logging.error('repeat file name. Impossible!')
    os.symlink(src_file, pathjoin(tar_dir, file_name))

# create all *_outlier folders
for abnormal_cls in classes[:3]:
    exp_name = '{}_outlier'.format(abnormal_cls)
    exp_folder = pathjoin(TAR_DIR, exp_name)
    vae_train_folder = pathjoin(exp_folder, 'vae_train')
    vae_test_folder = pathjoin(exp_folder, 'vae_test')

    # make folders
    os.mkdir(exp_folder)
    os.mkdir(vae_train_folder)
    os.mkdir(vae_test_folder)

    # vae_test_folder
    test_normal_folder = pathjoin(vae_test_folder, 'normal')
    test_abnormal_folder = pathjoin(vae_test_folder, 'abnormal')
    os.mkdir(test_normal_folder)
    os.mkdir(test_abnormal_folder)

    # vae_train_folder
    vae_train_train_folder = pathjoin(vae_train_folder, 'train')
    vae_train_val_folder = pathjoin(vae_train_folder, 'val')
    os.mkdir(vae_train_train_folder)
    os.mkdir(vae_train_val_folder)
    vae_train_train_normal_folder = pathjoin(vae_train_train_folder, 'normal')
    vae_train_val_normal_folder = pathjoin(vae_train_val_folder, 'normal')
    os.mkdir(vae_train_train_normal_folder)
    os.mkdir(vae_train_val_normal_folder)

    # fill in vae_test
    logging.info('filling in %s', vae_test_folder)
    for img_path in tqdm(img_test_paths):
        img_file_name = os.path.split(img_path)[1]
        img_cls = get_class(img_path, classes)

        # determine target folder in vae_test
        if img_cls == abnormal_cls:
            target_folder = test_abnormal_folder
        else:
            target_folder = test_normal_folder

        symlink_without_replace(src_file=img_path, tar_dir=target_folder)

    # fill in vae_train/val/normal
    logging.info('filling in %s', vae_train_val_normal_folder)
    for img_path in tqdm(img_val_paths):
        if get_class(img_path, classes) != abnormal_cls:
            symlink_without_replace(src_file=img_path, tar_dir=vae_train_val_normal_folder)

    # fill in vae_train/train/normal
    logging.info('filling in %s', vae_train_train_normal_folder)
    for img_path in tqdm(img_train_paths):
        if get_class(img_path, classes) != abnormal_cls:
            symlink_without_replace(src_file=img_path, tar_dir=vae_train_val_normal_folder)

