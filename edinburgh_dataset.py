import numpy as np
import cv2
import pandas as pd

images1 = np.load('Edinburgh mfEIT Dataset/img1.npy')
images2 = np.load('Edinburgh mfEIT Dataset/img2.npy')
images3 = np.load('Edinburgh mfEIT Dataset/img3.npy')

v1 = np.load('Edinburgh mfEIT Dataset/V1.npy')
v2 = np.load('Edinburgh mfEIT Dataset/V2.npy')
v3 = np.load('Edinburgh mfEIT Dataset/V3.npy')


print(images1.shape)
print(images2.shape)
print(images3.shape)

print(v1.shape)
print(v2.shape)
print(v3.shape)

def seperate_images(array):
    """
    Seperates the numpy array into all the images and returns a list of images
    :param array: numpy array of images
    :return:
    """
    all_images = []
    for i in range(0, len(array)):
        img = array[i]
        img = img * 255
        img = img.astype(np.uint8)
        img = -img
        # clip between 0 and 255
        img = np.clip(img, 0, 255)
        # separate into 4 images that are stacked vertically each image is 64x64 and is captured using a different frequency
        img1 = img[0:64, 0:64]
        img2 = img[64:128, 0:64]
        img3 = img[128:192, 0:64]
        img4 = img[192:256, 0:64]
        all_images.append(img1)
        all_images.append(img2)
        all_images.append(img3)
        all_images.append(img4)
        # cv2.imshow('img1', cv2.resize(img1, (256, 256)))
        # cv2.imshow('img2', cv2.resize(img2, (256, 256)))
        # cv2.imshow('img3', cv2.resize(img3, (256, 256)))
        # cv2.imshow('img4', cv2.resize(img4, (256, 256)))
        # cv2.waitKey(1)
    all_images = np.array(all_images)
    return all_images

def seperate_voltages(array):
    """
    Seperates the numpy array into all the voltages and returns a list of voltages
    :param array:
    :return:
    """
    all_volts = []
    for i in range(0, len(array)):
        volts = array[i]
        volt1 = volts[0]
        volt2 = volts[1]
        volt3 = volts[2]
        volt4 = volts[3]
        all_volts.append(volt1)
        all_volts.append(volt2)
        all_volts.append(volt3)
        all_volts.append(volt4)
    # convert to np array
    all_volts = np.array(all_volts)
    return all_volts


voltages1 = seperate_voltages(v1)
voltages2 = seperate_voltages(v2)
voltages3 = seperate_voltages(v3)
all_voltages = np.concatenate((voltages1, voltages2, voltages3), axis=0)
#
# save as npy
np.save('Edinburgh mfEIT Dataset/voltages.npy', all_voltages)
print(len(voltages1))
all_images1 = seperate_images(images1)
# all_images2 = seperate_images(images2)
# all_images3 = seperate_images(images3)
# all_images = np.concatenate((all_images1, all_images2, all_images3), axis=0)
# binarize the images
# all_images1 = np.where(all_images1 > 0, 1, 0)
# save as npy
np.save('Edinburgh mfEIT Dataset/images1.npy', all_images1)
print(len(all_images1))
print("OK")

# voltages2 = seperate_voltages(v2)
# print(len(voltages2))
# all_images2 = seperate_images(images2)
# print(len(all_images2))
#
# voltages3 = seperate_voltages(v3)
# print(len(voltages3))
# all_images3 = seperate_images(images3)
# print(len(all_images3))


for img in all_images1:
    cv2.imshow('img1', cv2.resize(img, (256, 256)))
    cv2.waitKey(1)


# Create a dataframe with the images and voltages

df = pd.DataFrame(list(zip(all_images1, voltages1)), columns=['Images', 'Voltages'])

# export as pickle
df.to_pickle('Edinburgh mfEIT Dataset/df.pkl')
# crete a cnn model that takes in voltages and outputs the image
# shape of input is 104x1
# shape of output is 64x64x1






