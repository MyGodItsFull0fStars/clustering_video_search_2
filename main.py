from typing import List

from keyframe_clustering import KeyFrameClustering, Cluster, ClusteringType
from html_generator import HTMLGenerator

import utils


def create_html(clustering: ClusteringType):
    global kfc
    global generator
    global high_keyframes
    cluster_list: List[Cluster] = kfc.hierarchical_clustering(clustering)
    all_points_clusters = [kfc.cluster_list_150, kfc.cluster_list_50, kfc.cluster_list_30, kfc.cluster_list_10,
                           kfc.cluster_list_5, cluster_list]
    generator = HTMLGenerator(high_keyframes)

    for cluster in all_points_clusters:
        generator.create_html_with_cluster(clustering.name, f'{clustering.name}_{len(cluster)}_clusters',
                                           cluster)


if __name__ == '__main__':
    high_keyframes: bool = True
    kfc = KeyFrameClustering(high_keyframes)
    generator = HTMLGenerator(high_keyframes)

    clustering_type_list = list(map(ClusteringType, ClusteringType))
    
    for clustering_type in clustering_type_list:
        create_html(clustering_type)
