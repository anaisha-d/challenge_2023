import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.color import rgb2gray
from skimage import util
from skimage.filters import gaussian
#data loading
def load_data(path):
    with np.load(path) as data:
        train_images = data["train_images"]
        val_images = data["val_images"]
        test_images = data["test_images"]
    print("shape", train_images[0].shape)
    return train_images, val_images, test_images

#Used scikit-image documentation: https://scikit-image.org/docs/stable/auto_examples/index.html

def grayscale(image):
    plt.imshow(image, cmap='gray')
    plt.savefig('images/Anaisha_Das_1.png')
    plt.show()
    return image

def invert_contrast(image):
    
    invert_filter = np.full(image.shape,255)
    invert_image = invert_filter - image
    invert_image = invert_image.astype(np.uint8)
   
    plt.imshow(invert_image, cmap='gray')
    plt.savefig('images/Anaisha_Das_2.png')
    plt.show()
    return image

#Prompt 3: gaussian blur
def gaussian_blur(image):
    gaussian_img = gaussian(image, sigma=1, truncate=3.5, channel_axis=2)
    plt.imshow(gaussian_img, cmap='gray')
    plt.savefig('images/Anaisha_Das_3.png')
    plt.show()
    return gaussian_img


train_images, val_images, test_images = load_data('/Users/sdas/ucsfci2/input/chestmnist.npz')
img = train_images[0]
grayscale(img)
invert_contrast(img)
gaussian_blur(img)