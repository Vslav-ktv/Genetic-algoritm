import random
import math
from tkinter import *
import numpy as np
from PIL import Image, ImageTk, ImageFilter
from deap import creator, base, tools
from matplotlib import pyplot as plt

filename_1 = "map.png"
with Image.open(filename_1) as map_:
    map_.load()
filename_2 = "toFind.png"
with Image.open(filename_2) as toFind:
    toFind.load()
WIDTH, HEIGHT = map_.size
width, height = toFind.size
AVG_min = 10  # %
POPULATION_SIZE = 10000
P_CROSSOVER = 0.9
P_MUTATION = 0.1
MAX_GENERATIONS = 1000


# img_gray = map_.convert("L")
# img_gray_smooth = img_gray.filter(ImageFilter.SMOOTH)
# edges_smooth = img_gray_smooth.filter(ImageFilter.FIND_EDGES)
# contour = img_gray_smooth.filter(ImageFilter.CONTOUR)
# map_ = contour.point(lambda x: 0 if x == 255 else 255)
# img_gray = toFind.convert("L")
# img_gray_smooth = img_gray.filter(ImageFilter.SMOOTH)
# edges_smooth = img_gray_smooth.filter(ImageFilter.FIND_EDGES)
# contour = img_gray_smooth.filter(ImageFilter.CONTOUR)
# toFind = contour.point(lambda x: 0 if x == 255 else 255)


def create_(w, h, width, height):
    w1 = w + width
    h1 = h + height
    return w, h, w1, h1


def create_individual():
    return [random.randint(0, WIDTH - width), random.randint(0, HEIGHT - height)]


def cx_wh(ind1, ind2):
    ind1[0][0], ind2[0][1] = ind2[0][0], ind1[0][1]
    return


def Fitness(ind):
    img_ = map_.crop(create_(ind[0][0], ind[0][1], width, height))
    diff = np.abs(np.array(toFind) - np.array(img_))
    fitness = int(np.sum(diff))
    return fitness,


def mutInRange(ind, indpb):
    for i in range(len(ind)):
        if random.randint(0, 100) < indpb:
            ind[0][0] = random.randint(0, WIDTH - width)
            ind[0][1] = random.randint(0, HEIGHT - height)
    return ind

class Fw:
    def __init__(self, w, h):
        self.root = Tk()
        self.root.geometry(f'{w}x{h}')
        self.w = w
        self.h = h
        self.canvas = Canvas(self.root, width=w, height=h)
        self.canvas.pack()
        self.addImage(filename_1)
        self.rec = []


    def addImage(self, file):
        img = Image.open(file)
        image = ImageTk.PhotoImage(img)
        self.canvas.create_image(self.w // 2 + 1, self.h // 2 + 1, image=image)

    def move(self, target, x, y):
        self.canvas.move(target, x, y)

    def mainLoop(self):
        self.root.mainloop()

    def cr(self, x, y, tag="rec"):
        if tag == "rec":
            self.canvas.create_rectangle(x, y, x + width, y + height, fill='', outline="red", tags="rec")
        elif tag == "best":
            self.canvas.create_rectangle(x, y, x + width, y + height, fill='', outline="purple", tags="best")

    def update(self):
        self.root.update()

    def delete_all_figures(self, tag="rec"):
        self.canvas.delete(tag)


def main():
    fwk = Fw(WIDTH, HEIGHT)
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register('select', tools.selLexicase, k=10)
    toolbox.register('mate', cx_wh)
    toolbox.register('mutate', mutInRange, indpb=90)
    toolbox.register("individualCreator", tools.initRepeat, creator.Individual, create_individual, 1)
    toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)
    toolbox.register("evaluate", Fitness)
    population = toolbox.populationCreator(n=POPULATION_SIZE)
    generationCounter = 0
    fitnessValues = list(map(toolbox.evaluate, population))
    for individual, fitnessValue in zip(population, fitnessValues):
        individual.fitness.values = fitnessValue
    fitnessValues = [individual.fitness.values[0] for individual in population]
    minFitnessValues = []
    meanFitnessValues = []
    minFitness = sum(np.array(toFind))
    while generationCounter < MAX_GENERATIONS:
        generationCounter = generationCounter + 1
        offspring = toolbox.select(population)
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < P_CROSSOVER:
                toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
        for mutant in offspring:
            if random.random() < P_MUTATION:
                toolbox.mutate(mutant)
            del mutant.fitness.values
        freshIndividuals = [ind for ind in offspring if not ind.fitness.valid]
        freshFitnessValues = list(map(toolbox.evaluate, freshIndividuals))
        for individual, fitnessValue in zip(freshIndividuals, freshFitnessValues):
            individual.fitness.values = fitnessValue
        population[:] = offspring
        fitnessValues = [ind.fitness.values[0] for ind in population]
        minFitness = min(fitnessValues)
        meanFitness = math.sqrt(sum([i ** 2 for i in fitnessValues]) / len(population))
        minFitnessValues.append(minFitness)
        meanFitnessValues.append(meanFitness)
        print("- Поколение {}: Мин приспособ. = {}, Средняя приспособ. = {}"
              .format(generationCounter, minFitness, meanFitness))
        best_index = fitnessValues.index(min(fitnessValues))
        best_ind = population[best_index]
        fwk.cr(best_ind[0][0], best_ind[0][1], "best")
        for i in population:
            fwk.cr(i[0][0], i[0][1])
        fwk.update()
        fwk.delete_all_figures()
    fwk.delete_all_figures("best")
    best_index = fitnessValues.index(min(fitnessValues))
    best_ind = population[best_index]
    fwk.cr(best_ind[0][0], best_ind[0][1], "best")
    plt.plot(minFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel("Поколение")
    plt.ylabel('Мин/средняя приспособленность')
    plt.title('Зависимость минимальной и средней приспособленности от поколения')
    plt.show()


main()
