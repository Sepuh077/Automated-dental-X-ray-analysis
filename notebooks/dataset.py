import os
import yaml
import shutil

from helper import (
    make_gray, 
    save_image_array, 
    get_bndbox_info, 
    get_xml_info, 
    get_center_by_percent, 
    get_tooth_classes, 
)
from augmentation import blur_image


class DataSet:
    def __init__(self, args):
        self.normalize_arguments(args)
        self.img_ext = ['jpg', 'png', 'jpeg']

        self.data_images_path = os.path.join(self.data_path, 'images')
        self.xmls_path = os.path.join(self.data_path, 'xmls')

        self.data_labels_path = os.path.join(self.data_path, 'labels')
        if not os.path.exists(self.data_labels_path):
            os.mkdir(self.data_labels_path)

        self.images_path = os.path.join(self.dataset_path, 'images')
        self.labels_path = os.path.join(self.dataset_path, 'labels')

        self.tooth_classes = get_tooth_classes()

    def normalize_arguments(self, args):
        self.data_path = args.data_path
        self.dataset_path = args.dataset
        if self.dataset_path[-1] == '/' or self.dataset_path[-1] == '\\':
            self.dataset_path = self.dataset_path[:-1]

        self.name = args.name if args.name else self.dataset_path.split('/')[-1].split('\\')[-1]

    def create_yaml_file(self):
        classes = []
        for i in range(1, 5):
            for j in range(1, 9):
                classes.append(f'{i}{j}')
        yaml_dict = {
                        'train': os.path.join('..', self.dataset_path, 'images', 'train').replace('\\', '/'),
                        'test': os.path.join('..', self.dataset_path, 'images', 'test').replace('\\', '/'),
                        'val': os.path.join('..', self.dataset_path, 'images', 'val').replace('\\', '/'),

                        'nc': 32,
                        'names': { i: v for i, v in enumerate(classes) },
                    }

        with open( os.path.join(self.dataset_path, self.name + '.yaml'), 'w' ) as yaml_file:
            yaml.dump(yaml_dict, yaml_file)

    def create_dirs_by_type(self, dir_type):
        image_dir_path = os.path.join(self.images_path, dir_type)
        label_dir_path = os.path.join(self.labels_path, dir_type)

        os.makedirs( image_dir_path, exist_ok=True )
        os.makedirs( label_dir_path, exist_ok=True )

        return image_dir_path, label_dir_path

    def move_data(self, image_path, label_path, image_dir_path, label_dir_path):
        image_name = image_path.split('/')[-1].replace('.png', '.jpg')
        label_name = label_path.split('/')[-1]

        new_image_path = os.path.join( image_dir_path, image_name )
        new_label_path = os.path.join( label_dir_path, label_name )

        img = make_gray(image_path)
        save_image_array(img, new_image_path)
        shutil.copy(label_path, new_label_path)

    def create_dirs_and_move_data(self, dir_type, image_paths, label_paths):
        image_dir_path, label_dir_path = self.create_dirs_by_type(dir_type)

        for i in range(len(image_paths)):
            self.move_data(image_paths[i], label_paths[i], image_dir_path, label_dir_path)

    def get_label_path(self, image_name):
        label_name = image_name.split('.')[0] + '.txt'
        label_path = os.path.join(self.data_labels_path, label_name)
        if os.path.exists(label_path):
            return label_path.replace('\\', '/')
        return None

    def get_all_data_paths(self):
        image_paths = []
        label_paths = []
        for file_name in os.listdir( self.data_images_path ):
            if file_name.split('.')[-1].lower() in self.img_ext:
                label_path = self.get_label_path(file_name)
                if not label_path:
                    continue
                image_paths.append( os.path.join(self.data_images_path, file_name).replace('\\', '/') )
                label_paths.append(label_path)

        return image_paths, label_paths

    def seperate_and_move_data(self, image_paths, label_paths):
        count = len(image_paths)
        end1 = int(0.8 * count)
        end2 = int(0.95 * count)

        self.create_dirs_and_move_data('train', image_paths[:end1], label_paths[:end1])
        self.create_dirs_and_move_data('test', image_paths[end1:end2], label_paths[end1:end2])
        self.create_dirs_and_move_data('val', image_paths[end2:], label_paths[end2:])


    def get_object_text(self, obj, width, height):
        class_name, xmin, ymin, xmax, ymax = get_bndbox_info(obj)

        xcenter, ycenter = get_center_by_percent(xmin, ymin, xmax, ymax, width, height)

        w = (xmax - xmin) / width
        h = (ymax - ymin) / height

        return f'{self.tooth_classes[int(class_name)]} {xcenter} {ycenter} {w} {h}\n'

    def xml_to_txt(self, xml_file_name):
        self.classes = []
        self.nc = 0

        xml_file_path = os.path.join(self.xmls_path, xml_file_name)
        txt_file_path = os.path.join(self.data_labels_path, xml_file_name.replace('.xml', '.txt'))

        _, width, height, objs = get_xml_info(xml_file_path)

        txt_data = []

        for obj in objs:
            txt_data.append( self.get_object_text(obj, width, height) )
        
        with open(txt_file_path, 'w') as file:
            file.writelines(txt_data)

    def processing_xmls(self):
        for xml_file in os.listdir(self.xmls_path):
            self.xml_to_txt(xml_file)

    def add_data(self):
        for folder_type in os.listdir(self.images_path):
            images = os.path.join(self.images_path, folder_type)
            labels = os.path.join(self.labels_path, folder_type)
            for image_name in os.listdir( images ):
                file_name, ext = os.path.splitext(image_name)
                label_path = os.path.join(labels, f'{file_name}.txt')

                blur_image( os.path.join(images, image_name), (1, 20), os.path.join(images, f'{file_name}_1{ext}') )
                shutil.copy( label_path, os.path.join(labels, f'{file_name}_1.txt'))

                blur_image( os.path.join(images, image_name), (20, 20), os.path.join(images, f'{file_name}_2{ext}') )
                shutil.copy(label_path, os.path.join(labels, f'{file_name}_2.txt'))

                blur_image( os.path.join(images, image_name), (1, 1), os.path.join(images, f'{file_name}_3{ext}') )
                shutil.copy(label_path, os.path.join(labels, f'{file_name}_3.txt'))

    def create(self):
        if not os.path.exists(self.dataset_path):
            print('Path does not exist!')
            return None

        self.processing_xmls()

        image_paths, label_paths = self.get_all_data_paths()

        self.seperate_and_move_data(image_paths, label_paths)

        self.add_data()
        
        self.create_yaml_file()

        return True
