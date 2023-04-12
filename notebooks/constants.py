import os
from pathlib import Path


ROOT = Path( os.path.dirname(os.path.dirname(__file__)) )
MEDIA = ROOT / 'media'
MEDIA_IMAGES = MEDIA / 'images'
MEDIA_TEETH = MEDIA_IMAGES / 'teeth'
MEDIA_SINGLE_TOOTH = MEDIA_IMAGES / 'single_tooth'
