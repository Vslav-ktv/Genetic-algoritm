import random
import math

from PIL import Image
from deap import creator, base, tools
from genetic import crossover_function, mutation_function, create_individual, fitness_function
from vo import *
import default


def run_search(
        haystack_path,
        needle_path,
        show_function,
        parameters=Parameters(default.indtosel,
                              default.popsize,
                              default.maxgen,
                              default.pcros,
                              default.pmut,
                              default.mut)):
    data = load_data(haystack_path, needle_path)
    toolbox = init_toolbox(data, parameters)
    search(data, toolbox, show_function, parameters)


def load_data(haystack_path, needle_path):
    with Image.open(haystack_path) as haystack:
        haystack.load()
    hw, hh = haystack.size
    with Image.open(needle_path) as needle:
        needle.load()
    nw, nh = needle.size
    return Data(haystack, hw, hh, needle, nw, nh)


def init_toolbox(data, parameters):
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register('select', tools.selLexicase, k=parameters.individuals_to_select)
    toolbox.register('mate', crossover_function)
    toolbox.register('mutate', mutation_function, indpb=parameters.p_mutation, data=data, parameters=parameters)
    toolbox.register("individualCreator", tools.initRepeat, creator.Individual,
                     lambda: create_individual(data), 1)
    toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)
    toolbox.register("evaluate", fitness_function, data=data)
    return toolbox


def search(data, toolbox, show_function, parameters):
    generationCounter = -1
    results = Results(finished=False, minFitnessValues=[], meanFitnessValues=[], bestIndividuals=[],
                      generationIndex=generationCounter, population=[])
    while generationCounter < parameters.max_generations:
        generationCounter += 1
        results.generationIndex = generationCounter
        results.population = next_generation(results.population, toolbox, parameters) if generationCounter != 0 \
            else first_generation(toolbox, parameters)
        update_results(results)
        report_generation(data, results, show_function)
    results.finished = True
    report_total(data, results, show_function)


def first_generation(toolbox, parameters):
    population = toolbox.populationCreator(n=parameters.population_size)
    fitnessValues = list(map(toolbox.evaluate, population))
    for individual, fitnessValue in zip(population, fitnessValues):
        individual.fitness.values = fitnessValue
    return population


def next_generation(population, toolbox, parameters):
    offspring = select_best(population, toolbox)
    crossover(offspring, toolbox, parameters)
    mutate(offspring, toolbox, parameters)
    offspring = offspring + toolbox.populationCreator(n=parameters.population_size - parameters.individuals_to_select)
    evaluate_changed(offspring, toolbox)
    population[:] = offspring
    return population


def select_best(population, toolbox):
    offspring = toolbox.select(population)
    offspring = list(map(toolbox.clone, offspring))
    return offspring


def crossover(offspring, toolbox, parameters):
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < parameters.p_crossover:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values


def mutate(offspring, toolbox, parameters):
    for mutant in offspring:
        if random.random() < parameters.p_mutation:
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
    results.bestIndividuals.append(results.population[fitnessValues.index(minFitness)])


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
