import random
import math
import numpy as np
from PIL import Image
from deap import creator, base, tools
from vo import Data, Results


INDIVIDUALS_TO_SELECT = 0
POPULATION_SIZE = 0
MAX_GENERATIONS = 0
P_CROSSOVER = 0
P_MUTATION = 0
MUTATION = 0


def crossover_function(ind1, ind2):
    mean_arifm_x = (ind1[0][0] + ind2[0][0]) // 2
    mean_arifm_y = (ind1[0][1] + ind2[0][1]) // 2
    mean_sqrt_x = math.sqrt((ind1[0][0] ** 2 + ind2[0][0] ** 2) // 2)
    mean_sqrt_y = math.sqrt((ind1[0][1] ** 2 + ind2[0][1] ** 2) // 2)
    ind1[0][0], ind1[0][1] = mean_arifm_x, mean_arifm_y
    ind2[0][0], ind2[0][1] = mean_sqrt_x, mean_sqrt_y
    return ind1, ind2


def mutation_function(individual, indpb, data):
    """This function applies mutation on the input individual.
        :param individual: Individual to be mutated.
        :param indpb: Independent probability for each attribute to be mutated.
        :returns: A tuple of one individual.
    """
    ind_x, ind_y = individual[0][0], individual[0][1]
    if random.random() < indpb:
        offset_x = random.randint(10, math.floor(data.n_width * MUTATION))
        ind_right = individual[0][0] + offset_x
        if ind_right < data.h_width:
            ind_x = ind_right
        else:
            ind_left = individual[0][0] - offset_x
            ind_x = ind_left
    if random.random() < indpb:
        offset_y = random.randint(10, math.floor(data.n_width * MUTATION))
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


def run_search(haystack_path, needle_path, show_function):
    init_global_variables()
    data = load_data(haystack_path, needle_path)
    toolbox = init_toolbox(data)
    search(data, toolbox, show_function)


def init_global_variables(
        indtosel=30,
        popsize=100,
        maxgen=1000,
        pcros=0.9,
        pmut=0.1,
        mut=0.1
        ):
    global INDIVIDUALS_TO_SELECT
    global POPULATION_SIZE
    global MAX_GENERATIONS
    global P_CROSSOVER
    global P_MUTATION
    global MUTATION
    INDIVIDUALS_TO_SELECT = indtosel
    POPULATION_SIZE = popsize
    MAX_GENERATIONS = maxgen
    P_CROSSOVER = pcros
    P_MUTATION = pmut
    MUTATION = mut


def load_data(haystack_path, needle_path):
    with Image.open(haystack_path) as haystack:
        haystack.load()
    hw, hh = haystack.size
    with Image.open(needle_path) as needle:
        needle.load()
    nw, nh = needle.size
    return Data(haystack, hw, hh, needle, nw, nh)


def init_toolbox(data):
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register('select', tools.selLexicase, k=INDIVIDUALS_TO_SELECT)
    toolbox.register('mate', crossover_function)
    toolbox.register('mutate', mutation_function, indpb=0.9, data=data)
    toolbox.register("individualCreator", tools.initRepeat, creator.Individual,
                     lambda: create_individual(data), 1)
    toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)
    toolbox.register("evaluate", fitness_function, data=data)
    return toolbox


def search(data, toolbox, show_function):
    generationCounter = -1
    results = Results(finished=False, minFitnessValues=[], meanFitnessValues=[],
                      generationIndex=generationCounter, population=None, bestIndex=0)
    while generationCounter < MAX_GENERATIONS:
        generationCounter += 1
        results.generationIndex = generationCounter
        results.population = next_generation(results.population, toolbox, data) if generationCounter != 0 \
            else first_generation(toolbox)
        update_results(results)
        report_generation(data, results, show_function)
    results.finished = True
    report_total(data, results, show_function)


def first_generation(toolbox):
    population = toolbox.populationCreator(n=POPULATION_SIZE)
    fitnessValues = list(map(toolbox.evaluate, population))
    for individual, fitnessValue in zip(population, fitnessValues):
        individual.fitness.values = fitnessValue
    return population


def next_generation(population, toolbox, data):
    offspring = select_best(population, toolbox)
    crossover(offspring, toolbox)
    mutate(offspring, toolbox)
    offspring = offspring + toolbox.populationCreator(n=POPULATION_SIZE - INDIVIDUALS_TO_SELECT)
    evaluate_changed(offspring, toolbox)
    population[:] = offspring
    return population


def select_best(population, toolbox):
    offspring = toolbox.select(population)
    offspring = list(map(toolbox.clone, offspring))
    return offspring


def crossover(offspring, toolbox):
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < P_CROSSOVER:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values


def mutate(offspring, toolbox):
    for mutant in offspring:
        if random.random() < P_MUTATION:
            toolbox.mutate(mutant)
            del mutant.fitness.values


def evaluate_changed(offspring, toolbox):
    freshIndividuals = [ind for ind in offspring if not ind.fitness.valid]
    freshFitnessValues = list(map(toolbox.evaluate, freshIndividuals))
    for individual, fitnessValue in zip(freshIndividuals, freshFitnessValues):
        individual.fitness.values = fitnessValue


def update_results(results):
    fitnessValues = [ind.fitness.values[0] for ind in results.population]
    minFitness = min(fitnessValues)
    meanFitness = math.sqrt(sum([i ** 2 for i in fitnessValues]) / len(results.population))
    results.minFitnessValues.append(minFitness)
    results.meanFitnessValues.append(meanFitness)
    results.best_index = fitnessValues.index(minFitness)


def report_generation(data, results, show_function):
    print("Generation {}: Total {}, Min={}, Mean={}"
          .format(results.generationIndex, len(results.population),
                  results.minFitnessValues[-1], results.meanFitnessValues[-1]))
    show_function(data, results)


def report_total(data, results, show_function):
    print("Completed. Min={}".format(min(results.minFitnessValues)))
    show_function(data, results)


if __name__ == '__main__':
    run_search("images/map.png", "images/toFind.png", lambda data, results:
               print("Generation {}: Total {}, Min={}, Mean={}"
                     .format(results.generationIndex, len(results.population),
                             results.minFitnessValues[-1], results.meanFitnessValues[-1]
                             )))
