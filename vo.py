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

    # generation
    generationIndex: int
    population: list
    bestIndex: int


@dataclass
class Parameters:
    individuals_to_select: int
    population_size: int
    MAX_GENERATIONS: int
    P_CROSSOVER: float
    P_MUTATION: float
    MUTATION: float
