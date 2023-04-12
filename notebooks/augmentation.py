import os
import cv2
import pandas as pd
from PIL import Image
from helper import get_xml_info, get_bndbox_info, get_center_by_percent, get_size_by_percent



def create_all_single_images():
    xml_path = '../data/full_teeth/xmls/'
    img_path = '../data/full_teeth/images/'
    single_teeth_img_path = '../data/single_tooth/images/'

    if not os.path.exists(single_teeth_img_path):
        os.mkdir(single_teeth_img_path)

    im = 1
    data = {
        'name': [],
        'class': [],
        'xcenter': [],
        'ycenter': [],
        'width': [],
        'height': [],
    }

    for file_name in os.listdir(img_path):
        name, ext = file_name.split('.')

        is_exists, w, h, objs = get_xml_info(xml_path + f'{name}.xml')

        if not is_exists:
            continue
        image = cv2.imread(img_path + file_name)
        for obj in objs:
            class_name, xmin, ymin, xmax, ymax = get_bndbox_info(obj)

            xcenter, ycenter = get_center_by_percent(xmin, ymin, xmax, ymax, w, h)
            width, height = get_size_by_percent(xmin, ymin, xmax, ymax, w, h)

            # xmin = max(0, xmin - 20)
            # ymin = max(0, ymin - 20)
            # xmax = min(w, xmax + 20)
            # ymax = min(h, ymax + 20)

            cv2.imwrite(single_teeth_img_path + f'{im}.jpg', image[ymin:ymax, xmin:xmax])
            data['name'].append( str(im) )
            data['class'].append(class_name)
            data['xcenter'].append(xcenter)
            data['ycenter'].append(ycenter)
            data['width'].append(width)
            data['height'].append(height)

            im += 1

    pd.DataFrame(data).to_csv('../data/single_tooth/data.csv', index=None, header=True)


def images_to_pdf(image_paths, file_name='1'):
    images = [Image.open(img_path).convert('RGB') for img_path in image_paths if os.path.exists(img_path)]

    images[0].save(f'{file_name}.pdf', save_all=True, append_images=images[1:])


def blur_image(image, kernel_size, save_path=None):
    if isinstance(image, str):
        image = cv2.imread(image)

    blured_image = cv2.blur(image, kernel_size)

    if save_path:
        cv2.imwrite(save_path, blured_image)

    return blured_image
