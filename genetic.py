import math
import random
import numpy as np


def crossover_function(ind1, ind2):
    mean_arifm_x = (ind1[0][0] + ind2[0][0]) // 2
    mean_arifm_y = (ind1[0][1] + ind2[0][1]) // 2
    mean_sqrt_x = math.sqrt((ind1[0][0] ** 2 + ind2[0][0] ** 2) // 2)
    mean_sqrt_y = math.sqrt((ind1[0][1] ** 2 + ind2[0][1] ** 2) // 2)
    ind1[0][0], ind1[0][1] = mean_arifm_x, mean_arifm_y
    ind2[0][0], ind2[0][1] = mean_sqrt_x, mean_sqrt_y
    return ind1, ind2


def mutation_function(individual, indpb, data, parameters):
    """This function applies mutation on the individual.
        :param individual: Individual to be mutated.
        :param indpb: Independent probability for each attribute to be mutated.
        :returns: A tuple of one individual.
    """
    ind_x, ind_y = individual[0][0], individual[0][1]
    if random.random() < indpb:
        offset_x = random.randint(1, math.floor(data.n_width * parameters.mutation))
        ind_right = individual[0][0] + offset_x
        if ind_right < data.h_width:
            ind_x = ind_right
        else:
            ind_left = individual[0][0] - offset_x
            ind_x = ind_left
    if random.random() < indpb:
        offset_y = random.randint(1, math.floor(data.n_width * parameters.mutation))
        ind_up = individual[0][1] + offset_y
        if ind_up < data.h_width:
            ind_y = ind_up
        else:
            ind_down = individual[0][1] - offset_y
            ind_y = ind_down
    individual[0][0], individual[0][1] = ind_x, ind_y
    return individual,


def create_individual(data):
    return [random.randint(0, data.h_width - data.n_width), random.randint(0, data.h_height - data.n_height)]


def get_coordinates(individual, width, height):
    return (individual[0][0], individual[0][1],
            individual[0][0] + width, individual[0][1] + height)


def fitness_function(individual, data):
    img = data.haystack.crop(get_coordinates(individual, data.n_width, data.n_height))
    diff = np.abs(np.array(data.needle) - np.array(img))
    fitness = int(np.sum(diff))
    return fitness,