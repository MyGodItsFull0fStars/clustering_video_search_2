from typing import List, Set

from dominate import document
from dominate.tags import *

from keyframe_clustering import Cluster

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

    def create_html_with_cluster(self, title: str = 'Photos', file_name: str = 'gallery',
                                 cluster_list: List[Cluster] = None) -> None:
        if cluster_list is None:
            return

        doc = document(title=title)
        with doc.head:
            link(rel='stylesheet', href='../style.css')

        with doc:
            for idx, cluster in enumerate(cluster_list):
                h1(f'Cluster {idx + 1}')
                with div(_class='row'):
                    keyframe_idxs: Set = {point.image_idx for point in cluster.point_list}
                    for kf_idx in keyframe_idxs:
                        image_path: str = self.file_names[kf_idx]
                        # div(img(src=f'{image_path}', _class='image'), _class='column')
                        div(img(src=f'../{image_path}', _class='image'), _class='column')

        with open(f'clusters/{file_name}.html', 'w') as file:
        # with open(f'{file_name}.html', 'w') as file:
            file.write(doc.render())
