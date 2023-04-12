import torch
import pickle
import cv2
import os
from bs4 import BeautifulSoup
from PIL import Image
import json
import pandas as pd
from random import randint


def from_percent_to_number(xmin, ymin, xmax, ymax, width, height):
    """
    This function gets coordinates of bounding boxes in percents and makes it in pixels.
    """
    return (
        int(xmin * height),
        int(ymin * width),
        int(xmax * height),
        int(ymax * width)
    )


def get_random_color():
    return (
            randint(0, 255),
            randint(0, 255),
            randint(0, 255)
        )


def load_model(path, map_location='cuda'):
    if not os.path.exists(path) or path.split('.')[-1] not in ['pt', 'pkl', 'pth']:
        raise ValueError('Wrong path!')
    
    if path.split('.')[-1] == 'pkl':
        with open(path, 'rb') as file:
            model = pickle.load(file)
    else:
        model = torch.load(path, map_location=map_location)

    return model


def get_tooth_classes():
    with open('../data/full_teeth/tooth_classes.json', 'r') as file:
        data = json.loads(file.read())

    return {
        int(key): int(value)
        for key, value in data.items()
    }


def get_tooth_classes_reverse():
    with open('../data/full_teeth/tooth_classes.json', 'r') as file:
        data = json.loads(file.read())

    return {
        int(value): int(key)
        for key, value in data.items()
    }


def draw_bndbox_on_image(image, class_name, start, end):
    color = get_random_color()
    image = cv2.rectangle(image, start, end, color, 2)
    if class_name:
        image = cv2.putText(
            image, 
            str(class_name), 
            (start[0] + 10, start[1] - 10), 
            fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
            fontScale=1, 
            color=color, 
            thickness=2)

    return image


def get_bndbox_info(obj):
    class_name = obj.find('name').text

    bnd_box = obj.find('bndbox')

    xmin = int(bnd_box.find('xmin').text)
    ymin = int(bnd_box.find('ymin').text)
    xmax = int(bnd_box.find('xmax').text)
    ymax = int(bnd_box.find('ymax').text)

    return class_name, xmin, ymin, xmax, ymax


def get_center_by_percent(xmin, ymin, xmax, ymax, width, height):
    return ( (xmin + xmax) / 2 / width, (ymin + ymax) / 2 / height )

def get_size_by_percent(xmin, ymin, xmax, ymax, width, height):
    return ( (xmax - xmin) / width, (ymax - ymin) / height )

def get_xml_info(xml_file_path):
    try:
        with open(xml_file_path, 'r') as f:
            data = f.read()
    except FileNotFoundError:
        return (False, None, None, None)

    soup = BeautifulSoup(data, 'xml')

    size = soup.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    objs = soup.find_all('object')

    return (True, width, height, objs)


def make_gray(image):
    if isinstance(image, str):
        image = cv2.imread(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) > 2 else image


def save_image_array(image, path):
    cv2.imwrite(path, image)


def get_bndbox_info_from_txt_row(row_txt, with_class_names=False):
    cls_index, xcenter, ycenter, width, height = [float(x) for x in row_txt.replace('\n', '').split()]

    cls_name = get_tooth_classes_reverse()[int(cls_index)]

    xmin = xcenter - width / 2
    xmax = xcenter + width / 2

    ymin = ycenter - height / 2
    ymax = ycenter + height / 2

    if with_class_names:
        return (int(cls_name), xmin, ymin, xmax, ymax)
    else:
        return (xmin, ymin, xmax, ymax)


def get_bndbox_infos_from_txt(txt_path, with_class_names=False):
    with open(txt_path, 'r') as file:
        rows = file.readlines()
    
    infos = []

    for row in rows:
        infos.append(get_bndbox_info_from_txt_row(row, with_class_names))

    return infos


def draw_rectangles(image_path):
    image_path = image_path.replace('\\', '/')
    file_name = image_path.split('/')[-1].split('.')[0]
    parent_dir = os.path.dirname( os.path.dirname(image_path) )
    label_path = os.path.join(parent_dir, 'labels', file_name + '.txt')
    save_path = os.path.join(parent_dir, 'checked', file_name + '.jpg')

    image = cv2.imread(image_path)

    width, height, *_ = image.shape

    bndboxes = get_bndbox_infos_from_txt(label_path, with_class_names=True)

    for bndbox_info in bndboxes:
        class_name, xmin, ymin, xmax, ymax = bndbox_info
        xmin, ymin, xmax, ymax = from_percent_to_number(xmin, ymin, xmax, ymax, width, height)
        image = draw_bndbox_on_image(image, str(class_name), (xmin, ymin), (xmax, ymax))

    image = cv2.putText(image, f'{file_name}.jpg', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return image
