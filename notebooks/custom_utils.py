import torch
import numpy as np
import os
import pandas as pd
import cv2
from image_processing import Processing
from helper import get_tooth_classes, make_gray, get_bndbox_infos_from_txt

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_single_tooth_data(image_shape=(200, 100)):
    """
    Loads images of single tooth and position
    """
    data_path = '../data/single_tooth/'
    images_path = data_path + 'images'

    classes = get_tooth_classes()

    images = []
    positions_data = []
    labels = []

    df = pd.read_csv(data_path + 'data.csv')

    data_count = df.shape[0]

    data_range = np.array( range(data_count), dtype=int )

    for i in data_range:
        info = df.iloc[i]
        images.append( cv2.imread( os.path.join(images_path, f'{int(info["name"])}.jpg') ) )
        positions_data.append( [info['ycenter'], info['xcenter']] )
        label = [0] * 32
        label[ classes[ info['class'] ] ] = 1
        labels.append(label)

    train_indxs = np.random.choice(data_range, int(data_count * 0.85), replace=False)
    test_indxs = np.delete( data_range, train_indxs )

    processing = Processing()
    processing.images = images
    images = torch.tensor( processing.modify_images( image_shape ) ).float().to(device)

    positions_data = torch.tensor( np.array(positions_data) ).float().to(device)
    labels = torch.tensor( np.array(labels) ).float().to(device)

    return ( 
        images[train_indxs], 
        positions_data[train_indxs], 
        labels[train_indxs], 

        images[test_indxs], 
        positions_data[test_indxs], 
        labels[test_indxs] 
    )


def single_tooth_as_input2(image_path, width, height, xcenter, ycenter):
    img = make_gray( cv2.imread( image_path ) )
        
    whole_width = int(img.shape[1] / width)
    whole_height = int(img.shape[0] / height)

    center_x = int(whole_width * xcenter)
    center_y = int(whole_height * ycenter)

    x_start = center_x - int(img.shape[1] / 2)
    x_end = x_start + img.shape[1]

    y_start = center_y - int(img.shape[0] / 2)
    y_end = y_start + img.shape[0]
    
    new_img = np.zeros((whole_height, whole_width), dtype=np.uint8)

    new_img[y_start:y_end, x_start:x_end] = img

    return new_img

def load_single_tooth_data_2(image_shape=(150, 400)):
    """
    Loads single tooth in big image(like full teeth) but other parts are 0
    """
    data_path = '../data/single_tooth/'
    images_path = data_path + 'images'

    classes = get_tooth_classes()

    images = []
    labels = []

    df = pd.read_csv(data_path + 'data.csv')

    data_count = df.shape[0]

    data_range = np.array( range(data_count), dtype=int )

    for i in data_range:
        info = df.iloc[i]
        
        img = single_tooth_as_input2(
            os.path.join(images_path, f'{int(info["name"])}.jpg'),
            width=info['width'],
            height=info['height'],
            xcenter=info['xcenter'],
            ycenter=info['ycenter']
        )
        
        images.append( img )
        
        label = [0] * 32
        label[ classes[ info['class'] ] ] = 1
        labels.append(label)

    train_indxs = np.random.choice(data_range, int(data_count * 0.85), replace=False)
    test_indxs = np.delete( data_range, train_indxs )

    processing = Processing()
    processing.images = images
    images = torch.tensor( processing.modify_images( image_shape ) ).float().to(device)

    labels = torch.tensor( np.array(labels) ).float().to(device)

    return ( 
        images[train_indxs], 
        labels[train_indxs], 

        images[test_indxs], 
        labels[test_indxs] 
    )


def load_full_teeth_data(image_shape=(400, 800)):
    data_path = '../data/full_teeth/'
    images_path = data_path + 'images'
    labels_path = data_path + 'labels'

    classes = get_tooth_classes()

    labels = []
    images = []

    for file_txt in os.listdir(labels_path):
        filename, ext = os.path.splitext(file_txt)

        # Label part
        infos = get_bndbox_infos_from_txt( os.path.join(labels_path, file_txt), with_class_names=True )

        out_array = np.random.random( (160,) )

        available_classes = []

        for info in infos:
            ind = classes[ info[0] ]
            available_classes.append(ind)

            out_array[ind * 5: ind * 5 + 5] = (1., *info[1:])

        for i in range(32):
            if i not in available_classes:
                out_array[i * 5] = 0.

        labels.append(out_array)
        
        # Input part
        images.append( cv2.imread( os.path.join(images_path, filename + '.jpg') ) )
    
    data_count = len(labels)

    data_range = np.array( range(data_count), dtype=int )

    train_indxs = np.random.choice(data_range, int(data_count * 0.8), replace=False)
    test_indxs = np.delete( data_range, train_indxs )

    processing = Processing()
    processing.images = images
    images = torch.tensor( processing.modify_images( image_shape ) ).float().to(device)

    labels = torch.tensor( np.array(labels) ).float().to(device)

    return (
        images[train_indxs],
        labels[train_indxs],

        images[test_indxs],
        labels[test_indxs]
    )
