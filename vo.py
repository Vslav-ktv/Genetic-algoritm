from PIL.ImageFile import ImageFile
from dataclasses import dataclass

@dataclass
class Data:
    haystack: ImageFile
    h_width: int
    h_height: int
    needle: ImageFile
    n_width: int
    n_height : int
