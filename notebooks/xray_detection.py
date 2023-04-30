import torch
import cv2
import shutil

from nets import Conv
from custom_utils import single_tooth_as_input2

from helper import get_bndbox_infos_from_txt, from_percent_to_number, get_tooth_classes_reverse, draw_bndbox_on_image

from yolov5.detect import run as yolov5_detect
from constants import *
from image_processing import Processing

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class XRayDetection:
    classes = get_tooth_classes_reverse()
    DISEASE_DETECTION_MODEL = torch.load(ROOT / 'main_models' / 'disease_cnn.pt', map_location=device)

    class ToothInfo:
        def __init__(self, xmin, ymin, xmax ,ymax, data, number=None):
            self.xmin = xmin
            self.ymin = ymin
            self.xmax = xmax
            self.ymax = ymax
            self.data = torch.tensor(data).to(device)
            self.number = number
        
        def set_number_by_index(self, index):
            self.number = XRayDetection.classes[index]

        def detect_disease(self):
            pass

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

    def create_input_data(self, label_path):
        self.single_images_path = MEDIA_SINGLE_TOOTH / self.filename

        self._create_folder(exist_ok=False)

        bndbox_infos = get_bndbox_infos_from_txt(label_path, with_class_names=True)

        image = cv2.imread(self.source)
        width, height, *_ = image.shape

        for bndbox_info in bndbox_infos:
            cls_name, xmin, ymin, xmax, ymax = bndbox_info

            x_min, y_min, x_max, y_max = from_percent_to_number(xmin, ymin, xmax, ymax, width, height)

            cv2.imwrite(self.single_images_path / f'{cls_name}.jpg', image[y_min:y_max, x_min:x_max])

            self.tooth_infos.append(
                XRayDetection.ToothInfo(
                    xmin=x_min,
                    ymin=y_min,
                    xmax=x_max,
                    ymax=y_max,
                    data = Processing.process_image(
                        image=image[y_min:y_max, x_min:x_max],
                        size=(150, 100),
                        gray=True
                    ),
                    number=cls_name
                )
                
            )

    def _collect_input_data(self):
        return torch.tensor(
            [info.data for info in self.tooth_infos]
        ).float().to(device)
    
    def generate_diagnosis(self, img_name, results):
        tooth_number = img_name.split('.')[0]
        if results.sum() == 0:
            return f'{tooth_number} համարի ատամը առողջ է։'
        
        text = ''
        plomb_text = ''
        
        if results[1] == 1:
            plomb_text = f'{tooth_number} համարի ատամը պլոմբավորված է:'
        if results[2] == 1:
            text = ('Այդ ' if plomb_text else f'{tooth_number} ') + 'ատամի վրա տեղադրված է շապիկ'
            if not results[3]:
                text += ':'
        if results[3] == 1:
            if text:
                text += ' և իմպլանտ։'
            else:
                text = ('Այդ ' if plomb_text else f'{tooth_number} ') + 'ատամի վրա տեղադրված է իմպլանտ:'
        
        if results[4]:
            if text or plomb_text:
                text += 'Ատամի նյարդը հեռացված է։'
            else:
                text = f'{tooth_number} համարի ատամի նյարդը հեռացված է։'
        if results[0]:
            if text or plomb_text:
                text += 'Ատամի մոտ նկատվել են այլ առողջական խնդիրներ։'
            else:
                text = f'{tooth_number} համարի ատամի մոտ նկատվել են այլ առողջական խնդիրներ։'

        return plomb_text + text
        
    
    def detect_image_disease(self, image, img_name):
        out = self.DISEASE_DETECTION_MODEL( torch.tensor(image).float().to(device)[None] )
        ans = torch.argmax(out, axis=-1).squeeze().detach().cpu().numpy()

        diagnosis_text = self.generate_diagnosis(img_name, ans)
        
        csv_text = f'{img_name},{ans[0]},{ans[1]},{ans[2]},{ans[3]},{ans[4]},{diagnosis_text}\n'

        return csv_text
        
    
    def detect_diseases(self):
        csv_texts = 'Անուն,Խնդիր,Պլոմբ,Շապիկ,Իմպլանտ,Նյարդ,Ախտորոշում\n'

        for img_name in os.listdir(self.single_images_path):
            img = cv2.imread( os.path.join(self.single_images_path, img_name) )
            img = Processing.process_image( img, (150, 100) ) / 255.
            csv_texts += self.detect_image_disease(img, img_name)
        
        with open( os.path.join(self.single_images_path, 'results.csv'), 'w' ) as file:
            file.write(csv_texts)

        print(csv_texts)

    def run(self):
        self._move_and_normalize_source()

        label_path = self._detect_teeth()

        self.create_input_data(label_path)

        self.detect_diseases()
    
if __name__ == '__main__':
    XRayDetection('/home/sepuh/workspace/diploma/data/full_teeth/images/37.jpg').run()
    
