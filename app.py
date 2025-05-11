import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk
from matplotlib import pyplot as plt
from vo import *
from search import *
import default

TAG_INDIVIDUAL = "individual"
TAG_BEST = "best"


class App:
    def __init__(self):
        # Window
        self.root = None
        self.canvas = None
        self.image = None

        # Entries
        self.e_haystack = None
        self.e_needle = None
        self.e_indtosel = None
        self.e_popsize = None
        self.e_maxgen = None
        self.e_pcros = None
        self.e_pmut = None
        self.e_mut = None

        # Labels
        self.l_indtosel = None
        self.l_popsize = None
        self.l_maxgen = None
        self.l_pcros = None
        self.l_pmut = None
        self.l_mut = None

        # Haystack size
        self.haystack_width = None
        self.haystack_height = None

    def start(self):
        self.create_window()
        self.root.mainloop()

    def create_window(self):
        self.root = tk.Tk()
        self.root.title("Genetic algorithm")
        self.create_upload_file_entries_and_buttons()
        self.create_labels()
        self.create_entries()
        bt_run = tk.Button(self.root, text="Run search", command=lambda: self.on_run_click())
        bt_run.grid(row=5, column=1)

    def create_upload_file_entries_and_buttons(self):
        self.e_haystack = tk.Entry(self.root, width=80)
        self.e_haystack.grid(row=1, column=0)
        bt_f1 = tk.Button(self.root, text="Choose haystack",
                          command=lambda: self.choose_file(self.e_haystack))
        bt_f1.grid(row=1, column=1)
        self.e_needle = tk.Entry(self.root, width=80)
        self.e_needle.grid(row=3, column=0)
        bt_f2 = tk.Button(self.root, text="Choose needle",
                          command=lambda: self.choose_file(self.e_needle))
        bt_f2.grid(row=3, column=1)

    def create_labels(self):
        self.l_indtosel = tk.Label(self.root, text="Individuals to select:")
        self.l_indtosel.grid(row=1, column=2)

        self.l_popsize = tk.Label(self.root, text="Population size:")
        self.l_popsize.grid(row=2, column=2)

        self.l_maxgen = tk.Label(self.root, text="Max generations:")
        self.l_maxgen.grid(row=3, column=2)

        self.l_pcros = tk.Label(self.root, text="Percent Crossover:")
        self.l_pcros.grid(row=4, column=2)

        self.l_pmut = tk.Label(self.root, text="Percent Mutation:")
        self.l_pmut.grid(row=5, column=2)

        self.l_mut = tk.Label(self.root, text="Mutation percent of needle size:")
        self.l_mut.grid(row=6, column=2)

    def create_entries(self):
        self.e_indtosel = tk.Entry(self.root, width=10)
        self.e_indtosel.grid(row=1, column=3)

        self.e_popsize = tk.Entry(self.root, width=10)
        self.e_popsize.grid(row=2, column=3)

        self.e_maxgen = tk.Entry(self.root, width=10)
        self.e_maxgen.grid(row=3, column=3)

        self.e_pcros = tk.Entry(self.root, width=10)
        self.e_pcros.grid(row=4, column=3)

        self.e_pmut = tk.Entry(self.root, width=10)
        self.e_pmut.grid(row=5, column=3)

        self.e_mut = tk.Entry(self.root, width=10)
        self.e_mut.grid(row=6, column=3)

    def choose_file(self, entry):
        file_path = filedialog.askopenfilename(title="Choose file")

        if file_path:
            entry.delete(0, tk.END)
            entry.insert(tk.END, file_path)

    def get_haystack_path(self) -> str:
        return self.e_haystack.get()

    def get_needle_path(self) -> str:
        return self.e_needle.get()

    def get_parameters(self) -> Parameters:
        indtosel = self.e_indtosel.get()
        popsize = self.e_popsize.get()
        maxgen = self.e_maxgen.get()
        pcros = self.e_pcros.get()
        pmut = self.e_pmut.get()
        mut = self.e_mut.get()
        if indtosel == "":
            indtosel = default.indtosel
        else:
            indtosel = int(indtosel)
        if popsize == "":
            popsize = default.popsize
        else:
            popsize = int(popsize)
        if maxgen == "":
            maxgen = default.maxgen
        else:
            maxgen = int(maxgen)
        if pcros == "":
            pcros = default.pcros
        else:
            pcros = int(pcros)
        if pmut == "":
            pmut = default.pmut
        else:
            pmut = int(pmut)
        if mut == "":
            mut = default.mut
        else:
            mut = int(mut)
        return Parameters(indtosel, popsize, maxgen, pcros, pmut, mut)

    def on_run_click(self):
        img = Image.open(self.get_haystack_path())
        self.image = ImageTk.PhotoImage(img)
        self.haystack_width, self.haystack_height = img.size
        self.canvas = tk.Canvas(self.root, width=self.haystack_width, height=self.haystack_height)
        self.canvas.create_image(self.haystack_width // 2 + 1, self.haystack_height // 2 + 1, image=self.image)
        self.canvas.grid()
        self.root.update()
        run_search(self.get_haystack_path(), self.get_needle_path(),
                   lambda d, r: self.show(d, r), self.get_parameters())

    def show(self, data: Data, results: Results):
        self.canvas.delete(TAG_INDIVIDUAL)
        self.canvas.delete(TAG_BEST)
        if results.finished:
            self.canvas.create_image(self.haystack_width // 2 + 1, self.haystack_height // 2 + 1, image=self.image)
            for i in range(len(results.bestIndividuals)):
                x = results.bestIndividuals[i][0][0]
                y = results.bestIndividuals[i][0][1]
                self.canvas.create_rectangle(x, y, x + data.n_width, y + data.n_height,
                                             fill='', outline="green", tags=TAG_BEST)
            # plt.plot(results.minFitnessValues, color='red')
            # plt.plot(results.meanFitnessValues, color='green')
            # plt.xlabel("Generation")
            # plt.ylabel('Min/mean fitness')
            # plt.title('Dependent min and mean fitness of generation')
            # plt.show()
        else:
            # for i in range(2): #len(results.population)
            #     x = results.population[i][0][0]
            #     y = results.population[i][0][1]
            #     self.canvas.create_rectangle(x, y, x + data.n_width, y + data.n_height,
            #                                  fill='', outline="blue", tags=TAG_INDIVIDUAL)
            best = results.bestIndividuals[-1]
            x = best[0][0]
            y = best[0][1]
            self.canvas.create_rectangle(x, y, x + data.n_width, y + data.n_height,
                                         fill='', outline="green", tags=TAG_BEST)
        self.root.update()


def run():
    app = App()
    app.start()


if __name__ == '__main__':
    run()
