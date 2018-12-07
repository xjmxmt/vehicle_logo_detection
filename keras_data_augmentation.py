import cv2
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os


def img_rename(path):
    dir = os.listdir(path)
    n = 0
    for each in dir:
        os.rename(os.path.join(path, each), os.path.join(path, str(n) + '.jpg'))
        n += 1


def img_show(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (272, 272))

    plt.imshow(img)
    img = np.transpose(img, (2, 0, 1))
    # print(img.shape)
    img = img.reshape((1,) + img.shape)
    # print(img.shape)
    return img


def aug_img(img, save_path):
    i = 0
    for batch in datagen.flow(img, batch_size=10,
                              save_to_dir=save_path, save_prefix='0',
                              save_format='jpeg'):
        # plt.subplot(5, 4, i + 1)
        # plt.axis('off')

        aug_img = batch[0]
        aug_img = aug_img.astype('float32')
        aug_img /= 255
        aug_img = np.transpose(aug_img, (1, 2, 0))
        # plt.imshow(aug_img)

        i += 1
        if i > 9: break


def tol_integrate_ops(tol_path, tol_save_path):
    dir = os.listdir(tol_path)
    for each in dir:
        separate_path = tol_path + each + '/'
        separate_save_path = tol_save_path + each + '/'
        # print(separate_path)
        print(separate_save_path)
        integate_ops(separate_path, separate_save_path)



def integate_ops(path, save_path):
    dir = os.listdir(path)
    for each in dir:
        img = img_show(path+each)
        if img is None: continue
        aug_img(img, save_path)
    img_rename(save_path)

# tol_path = 'F:/vehicle_detection/data/train/'
# tol_save_path = 'F:/vehicle_detection/data/train_aug2/'

path = 'F:/vehicle_detection/data/train_aug2/9/'
save_path = 'F:/vehicle_detection/data/train_aug2/9/'
datagen = ImageDataGenerator(
    zca_whitening=False,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0,
    zoom_range=0,
    horizontal_flip=True,
    fill_mode='nearest'
)
integate_ops(path, save_path)
# tol_integrate_ops(tol_path, tol_save_path)

# img = img_show(path)
# aug_img(img, save_path)
# img_rename(save_path)


