import random
import math
import numpy as np
from PIL import Image
from deap import creator, base, tools

POPULATION_SIZE = 10000
MAX_GENERATIONS = 1000
P_CROSSOVER = 0.9
P_MUTATION = 0.1

haystack_path = "images/map.png"
needle_path = "images/toFind.png"
with Image.open(haystack_path) as haystack:
    haystack.load()
WIDTH, HEIGHT = haystack.size
with Image.open(needle_path) as needle:
    needle.load()
width, height = needle.size


def crossover_function(ind1, ind2):
    ind1[0][0], ind2[0][1] = ind2[0][0], ind1[0][1]
    return


def mutation_function(individual, indpb):
    """This function applies mutation on the input individual.
        :param individual: Individual to be mutated.
        :param indpb: Independent probability for each attribute to be mutated.
        :returns: A tuple of one individual.
    """
    for i in range(len(individual)):
        if random.randint(0, 100) < indpb:
            individual[0][0] = random.randint(0, WIDTH - width)
            individual[0][1] = random.randint(0, HEIGHT - height)
    return individual,


def create_individual():
    return [random.randint(0, WIDTH - width), random.randint(0, HEIGHT - height)]


def get_coordinates(individual, width, height):
    return (individual[0][0], individual[0][1],
            individual[0][0] + width, individual[0][1] + height)


def fitness_function(individual):
    img = haystack.crop(get_coordinates(individual, width, height))
    diff = np.abs(np.array(needle) - np.array(img))
    fitness = int(np.sum(diff))
    return fitness,


def run():
    toolbox = init_toolbox()
    search(toolbox)
    pass


def init_toolbox():
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register('select', tools.selLexicase, k=10)
    toolbox.register('mate', crossover_function)
    toolbox.register('mutate', mutation_function, indpb=90)
    toolbox.register("individualCreator", tools.initRepeat, creator.Individual, create_individual, 1)
    toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)
    toolbox.register("evaluate", fitness_function)
    return toolbox


def search(toolbox):
    minFitnessValues = []
    meanFitnessValues = []
    generationCounter = -1
    while generationCounter < MAX_GENERATIONS:
        generationCounter += 1
        population = next_generation(population, toolbox) if generationCounter != 0 \
            else first_generation(toolbox)
        append_statistics(meanFitnessValues, minFitnessValues, population)
        report_generation(generationCounter, meanFitnessValues, minFitnessValues, population)
    report_total(minFitnessValues)


def first_generation(toolbox):
    population = toolbox.populationCreator(n=POPULATION_SIZE)
    fitnessValues = list(map(toolbox.evaluate, population))
    for individual, fitnessValue in zip(population, fitnessValues):
        individual.fitness.values = fitnessValue
    return population


def next_generation(population, toolbox):
    offspring = select_best(population, toolbox)
    crossover(offspring, toolbox)
    mutate(offspring, toolbox)
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


def append_statistics(meanFitnessValues, minFitnessValues, population):
    fitnessValues = [ind.fitness.values[0] for ind in population]
    minFitness = min(fitnessValues)
    meanFitness = math.sqrt(sum([i ** 2 for i in fitnessValues]) / len(population))
    minFitnessValues.append(minFitness)
    meanFitnessValues.append(meanFitness)


def report_generation(generationCounter, meanFitnessValues, minFitnessValues, population):
    print("Generation {}: Total {}, Min={}, Mean={}"
          .format(generationCounter, len(population), minFitnessValues[-1], meanFitnessValues[-1]))
    # best_index = fitnessValues.index(minFitnessValues[-1])
    #best_ind = population[best_index]
    #fwk.cr(best_ind[0][0], best_ind[0][1], "best")
    #for i in population:
    #    fwk.cr(i[0][0], i[0][1])
    #fwk.update()
    #fwk.delete_all_figures()

def report_total(minFitnessValues):
    print("Completed. Min={}"
          .format(min(minFitnessValues)))
    # fwk.delete_all_figures("best")
    # best_index = fitnessValues.index(min(fitnessValues))
    # best_ind = population[best_index]
    # fwk.cr(best_ind[0][0], best_ind[0][1], "best")
    # plt.plot(minFitnessValues, color='red')
    # plt.plot(meanFitnessValues, color='green')
    # plt.xlabel("Поколение")
    # plt.ylabel('Мин/средняя приспособленность')
    # plt.title('Зависимость минимальной и средней приспособленности от поколения')
    # plt.show()
    pass


if __name__ == '__main__':
    run()
