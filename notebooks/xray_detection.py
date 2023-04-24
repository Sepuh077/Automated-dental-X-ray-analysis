import torch
import cv2
import shutil

from nets import SingleToothNet, SingleToothNet2
from custom_utils import single_tooth_as_input2

from helper import get_bndbox_infos_from_txt, from_percent_to_number, get_tooth_classes_reverse, draw_bndbox_on_image

from yolov5.detect import run as yolov5_detect
from constants import *
from image_processing import Processing

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class XRayDetection:
    classes = get_tooth_classes_reverse()
    number_detection_model = torch.load(ROOT / 'models' / 'single_tooth_number_detection2.pt', map_location=device)

    class ToothInfo:
        def __init__(self, xmin, ymin, xmax ,ymax, data):
            self.xmin = xmin
            self.ymin = ymin
            self.xmax = xmax
            self.ymax = ymax
            self.data = data
            self.number = None
        
        def set_number_by_index(self, index):
            self.number = XRayDetection.classes[index]

        def draw_bndbox(self, image):
            image = draw_bndbox_on_image(
                image=image,
                class_name=self.number,
                start=(self.xmin, self.ymin),
                end=(self.xmax, self.ymax)
            )
            return image

    def __init__(self, source):
        self.yolov5_general_kwargs = {
            'exist_ok': True,
            'save_txt': True,
            'project': ROOT / 'results',
            'iou_thres': 0,
            'line_thickness': 2
        }

        self.source = source

        self.single_images_path = None
        self.tooth_infos = []

        self.get_filename_and_ext_from_source()

    def get_filename_and_ext_from_source(self):
        self.filename, self.ext = os.path.splitext( os.path.basename(self.source) )
        self.ext = self.ext[1:]

    def _create_folder(self, exist_ok=False):
        try:
            os.makedirs(self.single_images_path, exist_ok=exist_ok)
        except Exception:
            os.rmdir(self.single_images_path)
            self._create_folder(exist_ok)

    def _find_not_existed_path(self, path):
        filename, ext = '.'.join(str(path).split('.')[:-1]), str(path).split('.')[-1]

        ind = 1
        while os.path.exists(path):
            path = f'{filename}_{ind}.{ext}'
            ind += 1

        return path

    def _move_and_normalize_source(self):
        if not os.path.exists(self.source):
            raise FileNotFoundError(f'No such file {self.source}')

        parent_dir = os.path.dirname(self.source)

        if os.path.samefile(parent_dir, MEDIA_TEETH) and '.' not in self.filename:
            return self.source
        
        if '.' in self.filename:
            self.filename = self.filename.replace('.', '_')

        new_source = self._find_not_existed_path( MEDIA_TEETH / (self.filename + '.' + self.ext) )

        # shutil.move(self.source, new_source)
        shutil.copy(self.source, new_source)

        self.source = new_source
        self.get_filename_and_ext_from_source()

        return new_source

    def _detect_teeth(self):
        yolov5_detect(
            weights=ROOT / 'models' / 'full_teeth' / 'weights' / 'best.pt',
            source=self.source,
            name='full_teeth',
            **self.yolov5_general_kwargs
        )

        label_filename = self.filename + '.txt'

        label_path = ROOT / 'results' / 'full_teeth' / 'labels' / label_filename

        return label_path

    # def detect_tooth_number(self, image):
    #     pass

    def create_input_data(self, label_path):
        self.single_images_path = MEDIA_SINGLE_TOOTH / self.filename

        self._create_folder(exist_ok=False)

        bndbox_infos = get_bndbox_infos_from_txt(label_path, False)

        image = cv2.imread(self.source)
        width, height, *_ = image.shape

        ind = 1

        for bndbox_info in bndbox_infos:
            xmin, ymin, xmax, ymax = bndbox_info

            x_min, y_min, x_max, y_max = from_percent_to_number(xmin, ymin, xmax, ymax, width, height)

            cv2.imwrite(self.single_images_path / f'{ind}.jpg', image[y_min:y_max, x_min:x_max])

            self.tooth_infos.append(
                XRayDetection.ToothInfo(
                    xmin=x_min,
                    ymin=y_min,
                    xmax=x_max,
                    ymax=y_max,
                    data = Processing.process_image(
                        image=single_tooth_as_input2(
                            self.single_images_path / f'{ind}.jpg',
                            width=(xmax - xmin),
                            height=(ymax - ymin),
                            xcenter=(xmin + xmax) / 2,
                            ycenter=(ymin + ymax) / 2
                        ),
                        size=(150, 400),
                        gray=True
                    )
                )
                
            )
            ind += 1

    def _collect_input_data(self):
        return torch.tensor(
            [info.data for info in self.tooth_infos]
        ).float().to(device)
    
    def set_tooth_numbers(self, output):
        output = torch.argmax(output, dim=-1)
        for i, index in enumerate(output):
            self.tooth_infos[i].set_number_by_index(index.item())

    def detect_numbers(self):
        output = XRayDetection.number_detection_model(
            self._collect_input_data()
        )

        self.set_tooth_numbers(output)

    def get_image(self):
        image = cv2.imread(self.source)

        for info in self.tooth_infos:
            image = info.draw_bndbox(image)

        return image

    def run(self):
        self._move_and_normalize_source()

        label_path = self._detect_teeth()

        self.create_input_data(label_path)

        self.detect_numbers()

        return self.get_image()
    
if __name__ == '__main__':
    XRayDetection('/home/sepuh/workspace/diploma/data/full_teeth/images/250.jpg').run()
