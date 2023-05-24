import torch
import numpy as np
import os
import pandas as pd
import cv2
from image_processing import Processing
from helper import get_tooth_classes, make_gray, get_bndbox_infos_from_txt
from augmentation import blur_image

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

def load_diseases_data(image_shape=(150, 100)):
    """
    Loads single tooth in big image(like full teeth) but other parts are 0
    """

    data_path = '../data/single_tooth_disease/'
    images_path = data_path + 'images'

    images = []
    labels = []

    df = pd.read_csv(data_path + 'labels.csv')

    data_count = df.shape[0]

    for i in range(data_count):
        info = df.iloc[i]
        
        img = cv2.imread( os.path.join(images_path, info["name"] ) )
        
        images.append( img )

        label = [
            int(info['problem']),
            int(info['plomb']),
            int(info['shapik']),
            int(info['implant']),
            int(info['nerv']),
        ]

        for i in range(5):
            label[i] = [1, 0] if label[i] == 0 else [0, 1]
        
        labels.append(label)

        if label[3][1] == 1:
            images += [
                blur_image(img, (10, 10)),
                blur_image(img, (20, 20)),
                blur_image(img, (30, 10)),
                blur_image(img, (10, 30)),
                blur_image(img, (30, 30))
            ]
            labels += ([label] * 5)
        elif label[0][1] + label[1][1] + label[2][1] > 0:
            images += [
                blur_image(img, (10, 10)),
                blur_image(img, (20, 20)),
                blur_image(img, (30, 30))
            ]
            labels += ([label] * 3)

    data_count = len(labels)

    data_range = np.array( range(data_count), dtype=int )

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


def load_diseases_test_data(image_shape=(150, 100)):
    """
    Loads single tooth in big image(like full teeth) but other parts are 0
    """

    data_path = '../data/single_tooth_disease/'
    images_path = data_path + 'test_images'

    images = []
    labels = []

    df = pd.read_csv(data_path + 'test_labels.csv')

    data_count = df.shape[0]

    for i in range(data_count):
        info = df.iloc[i]
        
        img = cv2.imread( os.path.join(images_path, info["name"] ) )
        
        images.append( img )

        label = [
            int(info['problem']),
            int(info['plomb']),
            int(info['shapik']),
            int(info['implant']),
            int(info['nerv']),
        ]

        for i in range(5):
            label[i] = [1, 0] if label[i] == 0 else [0, 1]
        
        labels.append(label)

        # if label[3][1] == 1:
        #     images += [
        #         blur_image(img, (10, 10)),
        #         blur_image(img, (20, 20)),
        #         blur_image(img, (30, 10)),
        #         blur_image(img, (10, 30)),
        #         blur_image(img, (30, 30))
        #     ]
        #     labels += ([label] * 5)
        # elif label[0][1] + label[1][1] + label[2][1] > 0:
        #     images += [
        #         blur_image(img, (10, 10)),
        #         blur_image(img, (20, 20)),
        #         blur_image(img, (30, 30))
        #     ]
        #     labels += ([label] * 3)

    processing = Processing()
    processing.images = images
    images = torch.tensor( processing.modify_images( image_shape ) ).float().to(device)

    labels = torch.tensor( np.array(labels) ).float().to(device)

    return ( 
        images, 
        labels, 
    )
