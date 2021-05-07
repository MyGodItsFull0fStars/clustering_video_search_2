from keyframe_clustering import KeyFrameClustering, ClusterPoint, Cluster
import utils
import glob
import cv2
import numpy as np
from html_generator import HTMLGenerator
if __name__ == '__main__':
    kfc = KeyFrameClustering(True)
    
    cluster: Cluster = kfc.hierarchical_clustering()
    
    print(cluster.pop())
    #
    # print(len(kfc.cluster_list_150))
    # print(len(kfc.cluster_list_50))
    # print(len(kfc.cluster_list_30))
    # print(len(kfc.cluster_list_10))
    # print(len(kfc.cluster_list_5))

    # generator = HTMLGenerator(True)
    # generator.create_html()


    # c = Cluster()
    # a = Cluster()
    #
    # c_list = [c, a]
    #
    # print(c_list)
    # c_list.remove(a)
    # print(c_list)
    # file_name = 'asdf.jpg'
    #
    # image = cv2.imread(file_name)
    # histogram = utils.get_2D_histogram(image)
    # count = 0
    # for y, hist_row in enumerate(histogram):
    #     for x, value in enumerate(hist_row):
    #         if value > 0:
    #             count += 1
    #             print(f'({x}, {y}) -> {value}')
    # print(f'Count: {count}')
    # cluster = Cluster(0, image, 5)
    # print(cluster)



