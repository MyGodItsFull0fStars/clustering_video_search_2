from typing import List

from dominate import document
from dominate.tags import *

import utils


class HTMLGenerator:
    def __init__(self, high_keyframes: bool = False):
        self.file_names: List[str] = utils.get_key_frame_file_names(high_keyframes)

    def create_html(self, title: str = 'Photos'):
        with document(title=title) as doc:
            h1(title)
            for path in self.file_names:
                div(img(src=path), _class='photo')

        with open('gallery.html', 'w') as file:
            file.write(doc.render())
