from typing import Tuple, List

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
    K_1024_POINTS: str = 'k-1024-points'
    K_512_POINTS: str = 'k-512-points'
    K_128_POINTS: str = 'k-128-points'
    K_64_POINTS: str = 'k-64-points'
    K_32_POINTS: str = 'k-32-points'
    K_8_POINTS: str = 'k-8-points'


@unique
class SearchType(Enum):
    BRUTE_FORCE: str = 'brute-force'
    CLOSEST_PAIR_OF_PONTS: str = 'closest-pair-of-points'


@dataclass(order=True)
class ClusterPoint:
    image_idx: int
    value: int
    position: Tuple[int, int] = field(hash=True)  # position stored as (x,y) tuple


class Cluster:

    def __init__(self, sorted_point_list: bool = False):
        self.point_list: List[ClusterPoint] = []
        self.centroid: Tuple[int, int] = (-1, -1)
        self.sorted_point_list: bool = sorted_point_list
        self.is_dirty: bool = True

    def add_point(self, point: ClusterPoint) -> None:
        self.point_list.append(point)
        self.__sort_point_list()
        self.is_dirty = True

    def __sort_point_list(self):
        if self.sorted_point_list:
            self.point_list.sort()

    def add_point_list(self, points: List[ClusterPoint]) -> None:
        self.point_list.extend(points)
        self.__sort_point_list()
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

    def __str__(self):
        return f'Cluster -- ClusterPoints {[(point.position, point.value) for point in self.point_list]} -- Size {len(self.point_list)}'


class KeyFrameClustering:

    def __init__(self, high_keyframes: bool = False, search_type: SearchType = SearchType.BRUTE_FORCE):
        self._keyframes: list = utils.get_keyframe_image_list(high_keyframes)
        self.search_type: SearchType = search_type
        self.cluster_list_150: List[Cluster] = []
        self.cluster_list_50: List[Cluster] = []
        self.cluster_list_30: List[Cluster] = []
        self.cluster_list_10: List[Cluster] = []
        self.cluster_list_5: List[Cluster] = []

    @time_decorator
    def hierarchical_clustering(self, clustering_type: ClusteringType = ClusteringType.K_1024_POINTS):
        if not isinstance(clustering_type, ClusteringType):
            print('Given parameter {} is not a clustering type'.format(clustering_type))
            return None
        return self.merge_clusters(clustering_type=clustering_type)

    @time_decorator
    def merge_clusters(self, cluster_list: List[Cluster] = None,
                       clustering_type: ClusteringType = ClusteringType.K_1024_POINTS) -> \
            List[Cluster]:
        if cluster_list is None:
            cluster_list: List[Cluster] = self.__get_initial_cluster_list(clustering_type)

        cluster_list_length: int = len(cluster_list)

        if cluster_list_length <= 1:
            return cluster_list

        if cluster_list_length in [5, 10, 30, 50, 150]:
            self.__add_to_cluster_list(cluster_list)

        left_best, right_best, _ = self._find_best_clusters_brute_force(cluster_list)

        # merging points of clusters into left cluster
        cluster_list[left_best].add_point_list(cluster_list[right_best].point_list)
        # deleting right cluster after merge
        del cluster_list[right_best]

        return self.merge_clusters(cluster_list, clustering_type)

    def _find_best_clusters_brute_force(self, cluster_list) -> Tuple[int, int, float]:
        min_distance: float = np.inf
        left_best: int = 0
        right_best: int = 0

        for idx_left in range(len(cluster_list) - 1):
            left_cluster: Cluster = cluster_list[idx_left]
            left_centroid: Tuple[int, int] = left_cluster.get_centroid()
            for idx_right in range(idx_left + 1, len(cluster_list)):
                right_cluster: Cluster = cluster_list[idx_right]
                right_centroid: Tuple[int, int] = right_cluster.get_centroid()
                current_distance = self.__get_euclidian_distance(left_centroid, right_centroid)
                if current_distance < min_distance:
                    left_best = idx_left
                    right_best = idx_right
                    min_distance = current_distance

        return left_best, right_best, min_distance

    def _find_best_clusters_cpp(self, cluster_list: List[Cluster]) -> Tuple[int, int, float]:
        cluster_sorted_x: List[Cluster] = sorted(cluster_list, key=lambda x: x.point_list.position[0])
        cluster_sorted_y: List[Cluster] = sorted(cluster_list, key=lambda x: x.point_list.position[1])

        left_best, right_best, distance = self.closest_pair(cluster_sorted_x, cluster_sorted_y)

        return left_best, right_best, distance

    def closest_pair(self, cluster_sorted_x: List[Cluster], cluster_sorted_y: List[Cluster]) -> Tuple[int, int, float]:
        # TODO not yet implemented
        len_cluster_x: int = len(cluster_sorted_x)
        if len_cluster_x <= 3:
            return self._find_best_clusters_brute_force(cluster_sorted_x)
        mid: int = len_cluster_x // 2
        Qx = cluster_sorted_x[:mid]
        Rx = cluster_sorted_x[mid:]

        # Determine midpoint on x-axis
        # midpoint: int = cluster_sorted_x.[mid]
        pass

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

    def __get_initial_cluster_list(self, clustering_type: ClusteringType = ClusteringType.K_1024_POINTS) -> List[
        Cluster]:
        initial_cluster_list: List[Cluster] = []

        k_point_size: int = self.__get_k_size_with_clustering_type(clustering_type)

        for idx, image in enumerate(self._keyframes):
            initial_cluster_list.append(self.__get_cluster_with_k_points(idx, image, k_point_size))

        return initial_cluster_list

    def __get_k_size_with_clustering_type(self, clustering_type: ClusteringType) -> int:
        if clustering_type == ClusteringType.K_1024_POINTS:
            return 1024
        elif clustering_type == ClusteringType.K_512_POINTS:
            return 512
        elif clustering_type == ClusteringType.K_128_POINTS:
            return 128
        elif clustering_type == ClusteringType.K_64_POINTS:
            return 64
        elif clustering_type == ClusteringType.K_32_POINTS:
            return 32
        elif clustering_type == ClusteringType.K_8_POINTS:
            return 8

    def __get_cluster_with_k_points(self, image_idx, image, num_of_points: int = 1) -> Cluster:
        histogram = utils.get_2D_histogram(image)
        histogram = np.asarray(histogram)

        indices = (-histogram).argpartition(num_of_points, axis=None)[:num_of_points]
        temp_cluster: Cluster = Cluster()

        y_positions, x_positions = np.unravel_index(indices, histogram.shape)

        for i in range(len(x_positions)):
            x, y = x_positions[i], y_positions[i]
            value = histogram[y][x]
            point = ClusterPoint(image_idx, value, (x, y))
            temp_cluster.add_point(point)

        return temp_cluster
