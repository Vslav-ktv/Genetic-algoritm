import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk
from matplotlib import pyplot as plt

from search import *

TAG_INDIVIDUAL = "individual"
TAG_BEST = "best"


class App:
    def __init__(self):
        self.root = None
        self.e_haystack = None
        self.e_needle = None
        self.e_indtosel = None
        self.e_popsize = None
        self.e_maxgen = None
        self.e_pcros = None
        self.e_pmut = None
        self.e_mut = None
        self.canvas = None
        self.haystack_width = None
        self.haystack_height = None

    def start(self):
        self.create_window()
        self.root.mainloop()

    def create_window(self):
        self.root = tk.Tk()
        self.root.title("Genetic algoritm")
        self.e_haystack = tk.Entry(self.root, width=80)
        self.e_haystack.pack(padx=10, pady=10)
        bt_f1 = tk.Button(self.root, text="Choose haystack",
                          command=lambda: self.choose_file(self.e_haystack))
        bt_f1.pack(pady=10)
        self.e_needle = tk.Entry(self.root, width=80)
        self.e_needle.pack(padx=10, pady=20)
        bt_f2 = tk.Button(self.root, text="Choose needle",
                          command=lambda: self.choose_file(self.e_needle))
        self.e_indtosel = tk.Entry(self.root, width=10)
        self.e_indtosel.pack(padx=0, pady=0)

        self.e_popsize = tk.Entry(self.root, width=10)
        self.e_popsize.pack(padx=0, pady=0)

        self.e_maxgen = tk.Entry(self.root, width=10)
        self.e_maxgen.pack(padx=0, pady=0)

        self.e_pcros = tk.Entry(self.root, width=10)
        self.e_pcros.pack(padx=0, pady=0)

        self.e_pmut = tk.Entry(self.root, width=10)
        self.e_pmut.pack(padx=0, pady=0)

        self.e_mut = tk.Entry(self.root, width=10)
        self.e_mut.pack(padx=0, pady=0)
        bt_f2.pack(pady=20)
        bt_run = tk.Button(self.root, text="Run search", command=lambda: self.on_run_click())
        bt_run.pack(pady=30)

    def choose_file(self, entry):
        file_path = filedialog.askopenfilename(title="Choose file")

        if file_path:
            entry.delete(0, tk.END)
            entry.insert(tk.END, file_path)

    def get_haystack_path(self):
        return self.e_haystack.get()

    def get_needle_path(self):
        return self.e_needle.get()

    def on_run_click(self):
        img = Image.open(self.get_haystack_path())
        image = ImageTk.PhotoImage(img)
        self.haystack_width, self.haystack_height = img.size
        self.canvas = tk.Canvas(self.root, width=self.haystack_width, height=self.haystack_height)
        self.canvas.create_image(self.haystack_width // 2 + 1, self.haystack_height // 2 + 1, image=image)
        self.canvas.pack()
        self.root.update()
        run_search(self.get_haystack_path(), self.get_needle_path(),
                   lambda d, r: self.show(d, r))

    def show(self, data, results):
        if results.finished:
            # bestIndex = results.minFitnessValues.index(min(results.minFitnessValues))
            # best_ind = results.population[bestIndex]
            # x, y = best_ind[0][0], best_ind[0][1]
            # self.canvas.create_rectangle(x, y, x + data.n_width, y + data.n_height,
            #                              fill='', outline="purple", tags=TAG_BEST)
            # plt.plot(results.minFitnessValues, color='red')
            # plt.plot(results.meanFitnessValues, color='green')
            # plt.xlabel("Поколение")
            # plt.ylabel('Мин/средняя приспособленность')
            # plt.title('Зависимость минимальной и средней приспособленности от поколения')
            # plt.show()
            pass
        else:
            self.canvas.delete(TAG_INDIVIDUAL)
            self.canvas.delete(TAG_BEST)
            # for i in range(2): #len(results.population)
            #     x = results.population[i][0][0]
            #     y = results.population[i][0][1]
            #     self.canvas.create_rectangle(x, y, x + data.n_width, y + data.n_height,
            #                                  fill='', outline="blue", tags=TAG_INDIVIDUAL)
            best = results.population[results.bestIndex]
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
