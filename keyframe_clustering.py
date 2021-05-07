from typing import Tuple, List

import cv2
import glob
from decorators import time_decorator
from enum import Enum, unique
import numpy as np
from dataclasses import dataclass, field
from multiprocessing import Process, Value, Array, Pool, Lock
from copy import deepcopy

import utils


def add_to_cluster_list(lock, idx, image, cluster_list):
    histogram = utils.get_2D_histogram(image)
    temp_cluster: Cluster = Cluster()
    for y, hist_row in enumerate(histogram):
        for x, value in enumerate(hist_row):
            if value > 0:  # only consider positive values
                point: ClusterPoint = ClusterPoint(idx, value, (x, y))
                temp_cluster.add_point(point)
    lock.acquire()
    cluster_list.append(temp_cluster)
    lock.release()
    return temp_cluster


@unique  # enum values have to be unique
class ClusteringType(Enum):
    BRUTE_FORCE: str = 'brute-force'
    CPP: str = 'closest-pair-of-points'


@dataclass(order=True)
class ClusterPoint:
    image_idx: int
    value: int
    position: Tuple[int, int] = field(hash=True)  # position stored as (x,y) tuple


class Cluster:

    def __init__(self):
        self.point_list: List[ClusterPoint] = []
        self.centroid: Tuple[int, int] = (-1, -1)
        self.is_dirty: bool = True

    def add_point(self, point: ClusterPoint) -> None:
        self.point_list.append(point)
        self.point_list.sort()
        self.is_dirty = True

    def add_point_list(self, points: List[ClusterPoint]) -> None:
        self.point_list.extend(points)
        self.point_list.sort()
        self.is_dirty = True

    def get_centroid(self) -> Tuple[int, int]:
        if self.is_dirty:  # only calculate if new points were added
            # TODO maybe only use one loop
            x: List[int] = []
            y: List[int] = []
            for idx, point in enumerate(self.point_list):
                x.append(point.position[0])
                y.append(point.position[1])
            self.centroid = (sum(x) // len(self.point_list), sum(y) // len(self.point_list))
            self.is_dirty = False

        return self.centroid

    def add_cluster_points_from_image(self, image_idx, image, num_of_points: int = 1):
        histogram = utils.get_2D_histogram(image)
        histogram = np.asarray(histogram)

        indices = (-histogram).argpartition(num_of_points, axis=None)[:num_of_points]

        y_positions, x_positions = np.unravel_index(indices, histogram.shape)

        for i in range(len(x_positions)):
            x, y = x_positions[i], y_positions[i]
            value = histogram[y][x]
            point = ClusterPoint(image_idx, value, (x, y))
            self.add_point(point)

        self.is_dirty = True

    def __str__(self):
        return f'Cluster -- ClusterPoints {[(point.position, point.value) for point in self.point_list]} -- Size {len(self.point_list)}'


class KeyFrameClustering:

    def __init__(self, high_keyframes: bool = False):
        self._keyframes: list = utils.get_keyframe_image_list(high_keyframes)
        self.cluster_list_150: List[Cluster] = []
        self.cluster_list_50: List[Cluster] = []
        self.cluster_list_30: List[Cluster] = []
        self.cluster_list_10: List[Cluster] = []
        self.cluster_list_5: List[Cluster] = []

    @time_decorator
    def hierarchical_clustering(self, clustering_type: ClusteringType = ClusteringType.BRUTE_FORCE):
        if not isinstance(clustering_type, ClusteringType):
            print('Given parameter {} is not a clustering type'.format(clustering_type))
        if clustering_type == ClusteringType.BRUTE_FORCE:
            return self._brute_force_clustering()

        elif clustering_type == ClusteringType.CPP:
            pass

    @time_decorator
    def _brute_force_clustering(self, cluster_list: List[Cluster] = None) -> List[Cluster]:
        if cluster_list is None:
            cluster_list: List[Cluster] = self.__get_initial_cluster_list(ClusteringType.BRUTE_FORCE)

        cluster_list_length: int = len(cluster_list)

        if cluster_list_length <= 1:
            return cluster_list

        if cluster_list_length in [5, 10, 30, 50, 150]:
            self.__add_to_cluster_list(cluster_list)

        best_delta: float = np.inf
        left_best: int = 0
        right_best: int = 0

        for idx_left in range(len(cluster_list) - 1):
            left_cluster: Cluster = cluster_list[idx_left]
            left_centroid: Tuple[int, int] = left_cluster.get_centroid()
            for idx_right in range(idx_left + 1, len(cluster_list)):
                right_cluster: Cluster = cluster_list[idx_right]
                right_centroid: Tuple[int, int] = right_cluster.get_centroid()
                current_distance = self.__get_euclidian_distance(left_centroid, right_centroid)
                if current_distance < best_delta:
                    left_best = idx_left
                    right_best = idx_right
                    best_delta = current_distance

        # merging points of clusters into left cluster
        cluster_list[left_best].add_point_list(cluster_list[right_best].point_list)
        # deleting right cluster after merge
        del cluster_list[right_best]

        return self._brute_force_clustering(cluster_list)

    def __add_to_cluster_list(self, cluster_list):
        cluster_list_length: int = len(cluster_list)
        if cluster_list_length == 150:
            self.cluster_list_150 = deepcopy(cluster_list)
        elif cluster_list_length == 50:
            self.cluster_list_50 = deepcopy(cluster_list)
        elif cluster_list_length == 30:
            self.cluster_list_30 = deepcopy(cluster_list)
        elif cluster_list_length == 10:
            self.cluster_list_10 = deepcopy(cluster_list)
        elif cluster_list_length == 5:
            self.cluster_list_5 = deepcopy(cluster_list)

    def __get_euclidian_distance(self, left_position: Tuple[int, int], right_position: Tuple[int, int]) -> float:
        left_x, left_y = left_position[0], left_position[1]
        right_x, right_y = right_position[0], right_position[1]

        return np.sqrt(np.square(left_x - right_x) + np.square(left_y - right_y))

    def __get_initial_cluster_list(self, clustering_type: str = ClusteringType.BRUTE_FORCE) -> List[Cluster]:
        initial_cluster_list: List[Cluster] = []

        if clustering_type == ClusteringType.BRUTE_FORCE:
            # lock = Lock()
            # idxs = [i for i, _ in enumerate(self._keyframes)]
            # process = Process(target=add_to_cluster_list, args=(lock, idxs[0], self._keyframes[0], initial_cluster_list))
            # process.start()
            # process.join()

            # add_to_cluster_list()

            for idx, image in enumerate(self._keyframes):
                # for idx, image in enumerate(self._keyframes[:5]):
                histogram = utils.get_2D_histogram(image)
                temp_cluster: Cluster = Cluster()
                for y, hist_row in enumerate(histogram):
                    for x, value in enumerate(hist_row):
                        if value > 0:  # only consider positive values
                            point: ClusterPoint = ClusterPoint(idx, value, (x, y))
                            temp_cluster.add_point(point)
                initial_cluster_list.append(temp_cluster)

        return initial_cluster_list
