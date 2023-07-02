from PIL import Image
import numpy as np
from matplotlib import cm


def import_2Darray_from_image(filepath):
    pic = Image.open(filepath)
    # convert RGB colors to gray scale images by taking the mean of the color channels
    grayscale_image = np.array(pic.getdata()).reshape(pic.size[1], pic.size[0], -1).mean(axis=2) #mean is the mean of the 3rd dimension, so the mean of rbg meaning greys can be read
    # Get unique grayscale levels
    unique_levels = np.unique(grayscale_image)
    # Create a dictionary to map grayscale levels to class labels
    class_labels = {level: i for i, level in enumerate(unique_levels)}
    # Create an array of class labels corresponding to grayscale levels
    class_image = np.vectorize(lambda x: class_labels[x])(grayscale_image)
    return class_image


def plot_image_from_2Darray(normalized_data_array, color_map=cm.gist_earth):
    data_mapped = np.uint8(255 * color_map(normalized_data_array))
    img = Image.fromarray(data_mapped)
    img.show()


def save_image_from_2Darray(canvas, filepath, format='png'):
    data_mapped = np.uint8(255 * canvas)
    img = Image.fromarray(data_mapped)
    img.save(filepath, format=format)
