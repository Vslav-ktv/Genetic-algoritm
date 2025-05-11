import deap.base
from PIL.ImageFile import ImageFile
from dataclasses import dataclass


@dataclass
class Data:
    haystack: ImageFile
    h_width: int
    h_height: int
    needle: ImageFile
    n_width: int
    n_height: int


@dataclass
class Results:
    # total
    finished: bool
    minFitnessValues: list
    meanFitnessValues: list
    bestIndividuals: list
    # generation
    generationIndex: int
    population: list


@dataclass
class Parameters:
    individuals_to_select: int
    population_size: int
    max_generations: int
    p_crossover: float
    p_mutation: float
    mutation: float
