import os
import numpy as np
import cv2


class Processing:
    """
    This class will be used for processing images, which includes
    reshaping ( maintaining the same ration between sides ), adding black border and making gray.
    Parameters:
    path: 
        type: str or None
        description: if path is not None, from directory of this path will be loaded all images.
    """
    image_exts = ['jpg', 'jpeg', 'png', 'webp']

    def __init__(self, path=None):
        if path:
            self.images = Processing.load_dir(path)
        else:
            self.images = []

    def modify_images(self, shape):
        processed_images = []
        for image in self.images:
            processed_images.append( Processing.process_image(image, shape) )

        return np.array(processed_images) / 255.
    
    @staticmethod
    def load_dir(path):
        images = []
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isdir(file_path):
                dir_images = Processing.load_dir(file_path)
                images.extend(dir_images)
                del dir_images
            elif file_path.split('.')[-1] in Processing.image_exts:
                images.append( cv2.imread(file_path) )

        return images
    
    @staticmethod
    def add_border(image, new_shape):
        height, width = image.shape[:2]

        h, w = new_shape

        channels = image.shape[2] if len(image.shape) > 2 else 1

        if width < w:
            w1 = int( (w - width) / 2 )

            if w1 > 0:
                z1 = np.zeros( (height, w1, channels), dtype=np.uint8 ).squeeze(axis=-1)

                image = np.concatenate( (z1, image), axis=1 )
                
            w2 = w - width - w1
            z2 = np.zeros( (height, w2, channels), dtype=np.uint8 ).squeeze(axis=-1)

            image = np.concatenate( (image, z2), axis=1 )

        else:
            w = width
        
        if height < h:
            h1 = int( (h - height) / 2 )

            if h1 > 0:
                z1 = np.zeros( (h1, w, channels), dtype=np.uint8 ).squeeze(axis=-1)

                image = np.concatenate( (z1, image) )
            
            h2 = h - height - h1
            z2 = np.zeros( (h2, w, channels), dtype=np.uint8 ).squeeze(axis=-1)

            image = np.concatenate( (image, z2) )

        return image

    @staticmethod
    def reshape(image, new_shape):
        """
        This function reshapes image maintaining the same ration between sides
        Parameters:
        image:
            type: numpy.ndarray
        new_shape:
            type: iterable with 2 length
        """
        height, width = image.shape[:2]

        h, w = new_shape

        if h / height < w / width:
            return cv2.resize(image, ( int(width * h / height), h ) )
        else:
            return cv2.resize(image, ( w, int(height * w / width) ) )

    @staticmethod
    def make_gray(image):
        """
        This function returns gray image of given image
        (if given image is already gray this function returns same image)
        """
        if len(image.shape) > 2:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return image

    @staticmethod
    def process_image(image, size, gray=True):
        if isinstance(size, int):
            size = (size, size)
        
        if gray:
            image = Processing.make_gray(image)
        image = Processing.reshape(image, size)
        image = Processing.add_border(image, size)

        return image
