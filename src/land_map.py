import copy
import os
import random
from functools import partial

import numpy as np
from scipy.ndimage import measurements as measure

from src import logger
from src.image_io import import_2Darray_from_image

LOGGER = logger.get_logger()

DENSITY_LEVELS = ['high', 'medium', 'low']


class MapBlock:
    def __init__(self, x, y, inhabitants=0):
        self.x = x
        self.y = y
        self.is_nature = True
        self.is_built = False
        self.is_centrality = False
        self.inhabitants = inhabitants

    def set_block_population(self, block_population, density_level, population_density):
        self.inhabitants = block_population * population_density[density_level]
        self.density_level = density_level


class Land:
    def __init__(self, size_x, size_y, build_probability=0.5, neighboring_centrality_probability=5e-3,
                 isolated_centrality_probability=1e-1, T_star=5,
                 max_population=500000, max_ab_km2=10000, prob_distribution=(0.7, 0.3, 0),
                 density_factors=(1, 0.1, 0.01)):
        self.size_x = size_x
        self.size_y = size_y
        self.T_star = T_star

        self.map = [[MapBlock(x, y, inhabitants=0) for x in range(size_y)] for y in range(size_x)]

        self.centralities = {}
        self.inhabitants = {}
        # nature are the coordinates where we can safely build new buildings without checking for nature is reachable etc.

        self.central_points_candidates_1 = set()
        self.central_points_candidates_dict = {}

        for x in range(T_star, size_x - T_star):
            for y in range(T_star, size_y - T_star):
                self.central_points_candidates_1.add((x, y))
                self.central_points_candidates_dict[(x, y)] = True

        self.nature_dict = {}

        for x in range(size_x):
            for y in range(size_y):
                self.nature_dict[(x, y)] = True

        self.houses = {}
        self.neighbours = {}
        self.inhabitants = {}
        self.excluded = {}  # nature excluded from development

        self.build_probability = build_probability
        self.neighboring_centrality_probability = neighboring_centrality_probability
        self.isolated_centrality_probability = isolated_centrality_probability
        self.max_population = max_population
        # the assumption is that T_star is the number of blocks
        # that equals to a 15 minutes walk, i.e. roughly 1 km. 1 block has size 1000/T_star metres
        self.block_pop = max_ab_km2 / (T_star ** 2)
        self.probability_distribution = prob_distribution
        self.population_density = {'high': density_factors[0], 'medium': density_factors[1], 'low': density_factors[2],
                                   'empty': 0}

        self.avg_dist_from_nature = 0
        self.avg_dist_from_centr = 0
        self.max_dist_from_nature = 0
        self.max_dist_from_centr = 0
        self.current_population = 0
        self.current_centralities = 0
        self.current_built_blocks = 0
        self.current_free_nature = np.inf
        self.avg_dist_from_nature_wide = 0
        self.max_dist_from_nature_wide = 0

    @staticmethod
    def get_block_population(block_population, density_level, population_density):
        inhabitants = block_population * population_density[density_level]
        density_level = density_level
        return {'inhabitants': inhabitants, 'density_level': density_level}

    def check_consistency(self):
        for x in range(self.size_x):
            for y in range(self.size_y):
                block = self.map[x][y]
                assert (block.is_nature and not block.is_built) or (
                        block.is_built and not block.is_nature), f"({x},{y}) block has ambiguous coordinates"

    # def get_map_as_array(self):
    #     map_array = np.zeros(shape=(self.size_x, self.size_y))
    #     population_array = np.ones(shape=(self.size_x, self.size_y))
    #     for x in range(self.size_x):
    #         for y in range(self.size_y):
    #             if self.houses[(x, y)]:
    #                 map_array[x, y] = 1
    #             if self.centralities[(x, y)]:
    #                 map_array[x, y] = 2
    #             population_array[x, y] = self.inhabitants[(x, y)]
    #
    #     return map_array, population_array

    def add_neighbours(self, x, y):
        # print('add_neighbours', x, y)
        # print('add_neighbours2', self.neighbours)
        self.neighbours.pop((x, y), None)
        for i, j in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
            if 0 <= i < self.size_x and 0 <= j < self.size_y:
                self.neighbours[(i, j)] = True
        # print('add_neighbours2', self.neighbours)

    def set_centralities(self, centralities: list):
        for centrality in centralities:
            x, y = centrality.x, centrality.y
            self.nature_dict.pop((x, y))
            self.central_points_candidates_dict.pop((x, y))
            self.centralities[(x, y)] = True
            self.add_neighbours(x, y)

    def get_neighbors(self, x, y):
        neighbors = set()
        if x > 0:
            neighbors.add((x - 1, y))
        if x < (self.size_x - 1):
            neighbors.add((x + 1, y))
        if y > 0:
            neighbors.add((x, y - 1))
        if y < (self.size_y - 1):
            neighbors.add((x, y + y))
        return neighbors

    # used by classic simulation only
    def is_any_neighbor_built(self, x, y):
        assert self.T_star <= x <= self.size_x - self.T_star, f"point ({x},{y}) is not in the 'interior' of the land"
        assert self.T_star <= y <= self.size_y - self.T_star, f"point ({x},{y}) is not in the 'interior' of the land"
        return (self.map[x - 1][y].is_built or self.map[x + 1][y].is_built or self.map[x][y - 1].is_built or
                self.map[x][y + 1].is_built)

    def get_all_neighbors(self, coords):
        neighbors = set()
        for x, y in coords:
            neighbors.update(self.get_neighbors(x, y))
        return neighbors

    def get_all_nature_neighbors(self, coords):
        neighbors = self.get_all_neighbors(coords)
        result = set()
        for x, y in neighbors:
            if self.nature.get((x, y)):  # and not self.excluded.get((x, y)):
                result.add((x, y))
        return result

    def is_centrality_near(self, x, y):
        assert self.T_star <= x <= self.size_x - self.T_star, f"point ({x},{y}) is not in the 'interior' of the land"
        assert self.T_star <= y <= self.size_y - self.T_star, f"point ({x},{y}) is not in the 'interior' of the land"

        if len(self.centralities) < 4*self.T_star*self.T_star:
            for i, j in self.centralities:
                if d2(x, y, i, j) <= self.T_star * self.T_star:
                    return True
            return False
        else:
            for i in range(x - self.T_star, x + self.T_star + 1):
                for j in range(y - self.T_star, y + self.T_star + 1):
                    if self.centralities.get((i, j)):
                        if d2(x, y, i, j) <= self.T_star * self.T_star:
                            return True
            return False

    def nature_stays_extended(self, x, y):
        # this method assumes that x,y belongs to a natural region
        # land_array = np.ones([self.size_x, self.size_y])
        # for xx in range(0, self.size_x):
        #     for yy in range(0, self.size_y):
        #         if self.nature_dict.get((xx, yy)):
        #             land_array[xx, yy] = 0
        #
        # land_array[x, y] = 1
        #
        # nature_array = np.where(land_array == 0, 1, 0)
        # labels, num_features = measure.label(nature_array)

        if True: #num_features == 1:
            return self.nature_stays_wide(x, y)
        else:
            return 'ex'

    def nature_stays_wide(self, x, y):

        # along the horizontal axis, we traverse form the x coordinate of the test point right util we reach no nature no further than margin
        i = 1
        while self.nature_dict.get((x + i, y)) and i <= self.T_star and x + i < self.size_x:
            i += 1

        if self.T_star > i > 1:
            return 'no'

        # along the horizontal axis, we traverse form the x coordinate of the test point left util we reach no nature no further than margin
        i = 1
        while self.nature_dict.get((x - i, y)) and i <= self.T_star and x - i >= 0:
            i += 1

        if self.T_star > i > 1:
            return 'no'

        # along the vertical axis, we traverse form the y coordinate of the test point down util we reach no nature no further than margin
        i = 1
        while self.nature_dict.get((x, y + i)) and i <= self.T_star and y + i < self.size_y:
            i += 1

        if self.T_star > i > 1:
            return 'no'

        # along the vertical axis, we traverse form the y coordinate of the test point down util we reach no nature no further than margin
        i = 1
        while self.nature_dict.get((x, y - i)) and i <= self.T_star and y - i >= 0:
            i += 1

        if self.T_star > i > 1:
            return 'no'

        return 'yes'

    def nature_stays_reachable(self, x, y):
        for i in range(x - self.T_star, x + self.T_star + 1):
            for j in range(y - self.T_star, y + self.T_star + 1):
                if x != i and y != j and self.nature_dict.get((i, j)):
                    if d2(x, y, i, j) <= self.T_star * self.T_star:
                        return True
        return False

    def set_configuration_from_image(self, filepath):
        array_map = import_2Darray_from_image(filepath)
        for x in range(self.size_x):
            for y in range(self.size_y):
                if array_map[x, y] == 1:
                    self.centralities[(x, y)] = True
                    self.nature_dict.pop((x, y))
                    # TODO compute excluded nature
                    # self.map[x][y].is_built = True
                    # self.map[x][y].is_centrality = True
                    # self.map[x][y].is_nature = False

                if array_map[x, y] == 0:
                    self.houses[(x, y)] = True
                    self.nature_dict.pop((x, y))
                    # TODO compute excluded nature
                    # self.map[x][y].is_built = True
                    # self.map[x][y].is_centrality = False
                    # self.map[x][y].is_nature = False

    def set_current_counts(self, urbanism_model):
        self.current_population = np.array([x['inhabitants'] for x in self.inhabitants.values()]).sum()
        self.current_centralities = len(self.centralities)
        self.current_built_blocks = len(self.centralities) + len(self.houses)
        self.current_free_nature = len(self.nature_dict)
        tot_inhabited_blocks = len(self.houses)

        if tot_inhabited_blocks == 0:
            self.avg_dist_from_nature = 0
            self.avg_dist_from_centr = 0
            self.max_dist_from_nature = 0
            self.max_dist_from_centr = 0
        else:
            x_centr, y_centr = list(zip(*self.centralities.keys())) or [[], []]
            x_built, y_built = list(zip(*self.houses.keys())) or [[], []]
            distances_from_centr = np.sqrt(
                (np.array(x_built)[:, None] - x_centr) ** 2 + (np.array(y_built)[:, None] - y_centr) ** 2).min(
                axis=1)
            self.avg_dist_from_centr = distances_from_centr.sum() / tot_inhabited_blocks
            self.max_dist_from_centr = distances_from_centr.max()

            x_nature, y_nature = list(zip(*self.nature_dict.keys())) or [[], []]

            if urbanism_model == 'classical':
                nature_array = []  # np.where(land_array == 0, 1, 0)
                features, labels = 1, []  # measure.label(nature_array)
                unique, counts = np.unique(features, return_counts=True)
                large_natural_regions = counts[1:] >= self.T_star ** 2
                large_natural_regions_labels = unique[1:][large_natural_regions]
                x_nature_wide, y_nature_wide = np.where(np.isin(features, large_natural_regions_labels))
                distances_from_nature_wide = np.sqrt(
                    (x_built[:, None] - x_nature_wide) ** 2 + (y_built[:, None] - y_nature_wide) ** 2).min(
                    axis=1)
                self.avg_dist_from_nature_wide = distances_from_nature_wide.sum() / tot_inhabited_blocks
                self.max_dist_from_nature_wide = distances_from_nature_wide.max()

            distances_from_nature = np.sqrt(
                (np.array(x_built)[:, None] - x_nature) ** 2 + (np.array(y_built)[:, None] - y_nature) ** 2).min(
                axis=1)
            self.avg_dist_from_nature = distances_from_nature.sum() / tot_inhabited_blocks
            self.max_dist_from_nature = distances_from_nature.max()

    def set_record_counts_header(self, output_path, urbanism_model):
        filename = os.path.join(output_path, 'current_counts.csv')
        with open(filename, "a") as f:
            if urbanism_model == 'isobenefit':
                f.write(
                    "iteration,added_blocks,added_centralities,current_built_blocks,current_centralities,"
                    "current_free_nature,current_population,avg_dist_from_nature,avg_dist_from_centr,max_dist_from_nature,max_dist_from_centr\n")
            elif urbanism_model == 'classical':
                f.write(
                    "iteration,added_blocks,added_centralities,current_built_blocks,current_centralities,"
                    "current_free_nature,current_population,avg_dist_from_nature,avg_dist_from_wide_nature,"
                    "avg_dist_from_centr,max_dist_from_nature,max_dist_from_wide_nature,max_dist_from_centr\n")
            else:
                raise ValueError(
                    f"Invalid urbanism_model value: {urbanism_model}. Must be 'classical' or 'isobenefit'.")

    def record_current_counts(self, output_path, iteration, added_blocks, added_centralities, urbanism_model):
        filename = os.path.join(output_path, 'current_counts.csv')
        with open(filename, "a") as f:
            if urbanism_model == 'isobenefit':
                f.write(
                    f"{iteration},{added_blocks},{added_centralities},"
                    f"{self.current_built_blocks},{self.current_centralities},"
                    f"{self.current_free_nature},{self.current_population},"
                    f"{self.avg_dist_from_nature},{self.avg_dist_from_centr},"
                    f"{self.max_dist_from_nature},{self.max_dist_from_centr}\n")
            elif urbanism_model == 'classical':
                f.write(
                    f"{iteration},{added_blocks},{added_centralities},"
                    f"{self.current_built_blocks},{self.current_centralities},"
                    f"{self.current_free_nature},{self.current_population},"
                    f"{self.avg_dist_from_nature},{self.avg_dist_from_nature_wide},{self.avg_dist_from_centr},"
                    f"{self.max_dist_from_nature},{self.max_dist_from_nature_wide},{self.max_dist_from_centr}\n")
            else:
                raise ValueError(
                    f"Invalid urbanism_model value: {urbanism_model}. Must be 'classical' or 'isobenefit'.")


def d2(x1, y1, x2, y2):
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


def is_nature_wide_along_axis(array_1d, T_star):
    features, labels = measure.label(array_1d)
    unique, counts = np.unique(features, return_counts=True)
    if len(counts) > 1:
        return counts[1:].min() >= T_star
    else:
        return True


def central_points_count_to_place(centrality_probability, size_x, size_y):
    """
    central_points_count_to_place(centrality_probability, size_x, size_y)

        provides number of centralities to build on a given map

    """
    return np.random.binomial(n=size_x * size_y, p=(centrality_probability / (size_x * size_y)))


class IsobenefitScenario(Land):
    def update_map(self):
        added_blocks = 0
        added_centrality = self.place_central_points()

        for x, y in copy.deepcopy(self.neighbours):
            if np.random.rand() < self.build_probability:
                if self.can_build_house(x, y) == 'yes':
                    self.houses[(x, y)] = True
                    self.central_points_candidates_1.add((x, y))
                    # self.excluded[(x, y)] = True
                    self.nature_dict.pop((x, y), None)
                    density_level = np.random.choice(DENSITY_LEVELS, p=self.probability_distribution)
                    self.inhabitants[(x, y)] = Land.get_block_population(self.block_pop, density_level,
                                                                         self.population_density)
                    added_blocks += 1
                    # adding the neighbours of the new build hose to the global neighbours list
                    self.add_neighbours(x, y)

                    # else:
                    #     if np.random.rand() < self.isolated_centrality_probability / (self.size_x * self.size_y):
                    #         if self.nature_stays_extended(x, y):
                    #             if self.nature_stays_reachable(x, y):
                    #                 block.is_centrality = True
                    #                 block.is_built = True
                    #                 block.is_nature = False
                    #                 block.set_block_population(self.block_pop, 'empty', self.population_density)
                    #                 added_centrality += 1
        LOGGER.info(f"added blocks: {added_blocks}")
        LOGGER.info(f"added centralities: {added_centrality}")
        return added_blocks, added_centrality

    def place_central_points(self):
        central_points_count = central_points_count_to_place(self.isolated_centrality_probability, self.size_x,
                                                             self.size_y)
        points_placed = 0

        while self.central_points_candidates_1 and points_placed < central_points_count:

            x, y = random.sample(self.central_points_candidates_1, 1)[0]
            self.central_points_candidates_1.remove((x, y))
            if not self.nature_dict.get((x, y)):
                pass
            elif self.can_build(x, y) == 'yes':
                self.centralities[(x, y)] = True
                self.add_neighbours(x, y)
                # self.excluded[(x, y)] = True
                self.nature_dict.pop((x, y), None)
                self.inhabitants[(x, y)] = Land.get_block_population(self.block_pop, 'empty', self.population_density)
                points_placed += 1
            elif self.can_build(x, y) == 'ex':
                self.neighbours.pop((x, y), None)
                self.nature_dict.pop((x, y), None)
            elif self.can_build(x, y) == 'no':
                self.central_points_candidates_1.add((x, y))

        return points_placed

    def can_build(self, x, y):

        if self.nature_dict.get((x, y)):
            if not self.nature_stays_reachable(x, y):
                return 'ex'
            else:
                return self.nature_stays_extended(x, y)
        else:
            return 'ex'

    def can_build_house(self, x, y):
        if self.nature_dict.get((x, y)) and self.is_in_interior(x, y) and self.is_centrality_near(x, y):
            return self.can_build(x, y)
        else:
            return 'no'

    def is_in_interior(self, x, y):
        return self.T_star <= x < self.size_x - self.T_star and self.T_star <= y < self.size_y - self.T_star


class ClassicalScenario(Land):
    def is_any_neighbor_centrality(self, x, y):
        return (self.map[x - 1][y].is_centrality or self.map[x + 1][y].is_centrality or self.map[x][
            y - 1].is_centrality or
                self.map[x][y + 1].is_centrality)

    def update_map(self):
        added_blocks = 0
        added_centrality = 0
        copy_land = copy.deepcopy(self)
        for x in range(self.T_star, self.size_x - self.T_star):
            for y in range(self.T_star, self.size_y - self.T_star):
                block = self.map[x][y]
                assert (block.is_nature and not block.is_built) or (
                        block.is_built and not block.is_nature), f"({x},{y}) block has ambiguous coordinates"
                if block.is_nature:
                    if copy_land.is_any_neighbor_built(x, y):
                        if np.random.rand() < self.build_probability:
                            density_level = np.random.choice(DENSITY_LEVELS, p=self.probability_distribution)
                            block.is_nature = False
                            block.is_built = True
                            block.set_block_population(self.block_pop, density_level, self.population_density)
                            added_blocks += 1

                    else:
                        if np.random.rand() < self.isolated_centrality_probability / np.sqrt(
                                self.size_x * self.size_y) and (
                                self.current_built_blocks / self.current_centralities) > 100:
                            block.is_centrality = True
                            block.is_built = True
                            block.is_nature = False
                            block.set_block_population(self.block_pop, 'empty', self.population_density)
                            added_centrality += 1
                else:
                    if not block.is_centrality:
                        if block.density_level == 'low':
                            if np.random.rand() < 0.1:
                                block.set_block_population(self.block_pop, 'medium', self.population_density)
                        elif block.density_level == 'medium':
                            if np.random.rand() < 0.01:
                                block.set_block_population(self.block_pop, 'high', self.population_density)
                        elif block.density_level == 'high' and (
                                self.current_built_blocks / self.current_centralities) > 100:
                            if self.is_any_neighbor_centrality(x, y):
                                if np.random.rand() < self.neighboring_centrality_probability:
                                    block.is_centrality = True
                                    block.is_built = True
                                    block.is_nature = False
                                    block.set_block_population(self.block_pop, 'empty', self.population_density)
                                    added_centrality += 1
                            else:
                                if np.random.rand() < self.isolated_centrality_probability:  # /np.sqrt(self.current_built_blocks):
                                    block.is_centrality = True
                                    block.is_built = True
                                    block.is_nature = False
                                    block.set_block_population(self.block_pop, 'empty', self.population_density)
                                    added_centrality += 1

        LOGGER.info(f"added blocks: {added_blocks}")
        LOGGER.info(f"added centralities: {added_centrality}")
        return added_blocks, added_centrality
