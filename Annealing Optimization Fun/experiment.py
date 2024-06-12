import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import random
import math


def load_tsp(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    coords = []
    start_reading = False
    for line in lines:
        if start_reading:
            if line.strip() == 'EOF':
                break
            parts = line.strip().split()
            coords.append((float(parts[1]), float(parts[2])))
        if line.strip() == 'NODE_COORD_SECTION':
            start_reading = True
    return np.array(coords)


def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def calculate_distance_matrix(coords):
    num_cities = len(coords)
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            dist_matrix[i][j] = euclidean_distance(coords[i], coords[j])
    return dist_matrix


def objective_function(tour, dist_matrix):
    total_distance = 0
    num_cities = len(tour)
    for i in range(num_cities):
        total_distance += dist_matrix[tour[i]][tour[(i + 1) % num_cities]]
    return total_distance


def swap_cities(tour):
    new_tour = tour.copy()
    i, j = random.sample(range(len(tour)), 2)
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour


def reverse_subsection(tour):
    new_tour = tour.copy()
    i, j = sorted(random.sample(range(len(tour)), 2))
    new_tour[i:j + 1] = reversed(new_tour[i:j + 1])
    return new_tour


def insert_city(tour):
    new_tour = tour.copy()
    i, j = random.sample(range(len(tour)), 2)
    city = new_tour.pop(i)
    new_tour.insert(j, city)
    return new_tour


def chmiel_swap(tour):
    new_tour = tour.copy()
    i, j = sorted(random.sample(range(len(tour)), 2))
    subsection = new_tour[i:j + 1]

    for k in range(0, len(subsection) - 1, 2):
        subsection[k], subsection[k + 1] = subsection[k + 1], subsection[k]

    if len(subsection) % 2 == 1:
        last_index = len(subsection) - 1
        if last_index >= 2:
            subsection[last_index], subsection[last_index - 2] = subsection[last_index - 2], subsection[last_index]

    new_tour[i:j + 1] = subsection
    return new_tour


def simulated_annealing(coords, initial_temperature=100, cooling=0.95, num_epochs=200, iterations_per_epoch=100):
    dist_matrix = calculate_distance_matrix(coords)
    num_cities = len(coords)

    current_solution = list(range(num_cities))
    random.shuffle(current_solution)
    best_solution = current_solution
    best_fitness = objective_function(best_solution, dist_matrix)
    record_best_fitness = []
    record_solutions = []

    for epoch in range(num_epochs):
        current_temperature = initial_temperature * cooling ** epoch
        for _ in range(iterations_per_epoch):
            new_solution = random.choice([swap_cities, reverse_subsection, insert_city, chmiel_swap])(current_solution)
            current_fitness = objective_function(current_solution, dist_matrix)
            new_fitness = objective_function(new_solution, dist_matrix)

            if new_fitness < current_fitness or random.random() < math.exp(
                    (current_fitness - new_fitness) / current_temperature):
                current_solution = new_solution
                best_fitness = new_fitness
                if new_fitness < best_fitness:
                    best_solution = new_solution
                    best_fitness = new_fitness

            record_best_fitness.append(best_fitness)
        record_solutions.append(current_solution)

    return best_solution, best_fitness, record_best_fitness, record_solutions


def get_diff_result(problem, total_distance):
    optimal_distances = {
        "lin105.tsp": 14379,
        "tsp225.tsp": 3919,
        "pr1002.tsp": 259045,
        "pr2392.tsp": 378032,
        "rl5934.tsp": 556045
    }

    if problem in optimal_distances:
        optimal_distance = optimal_distances[problem]
        diff = ((total_distance / optimal_distance) - 1) * 100
        return f"{diff:.2f}%"
    else:
        return "Unknown problem"


def plot_tour(ax, coords, tour, title='TSP Tour'):
    ax.clear()
    ax.plot(coords[tour][:, 0], coords[tour][:, 1], 'o-', label='Tour')
    ax.plot([coords[tour[-1]][0], coords[tour[0]][0]], [coords[tour[-1]][1], coords[tour[0]][1]],
            'o-')  # Close the loop
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()


def plot_fitness(record_best_fitness, best_fitness, tsp_name):
    plt.figure()
    plt.plot(record_best_fitness)
    plt.title('Fitness changes over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Total Distance')

    diff = get_diff_result(tsp_name, best_fitness)
    plt.figtext(0.85, 0.25, f'Best Cost Tour: {best_fitness}\nPercentage from Optimal: {diff}',
                horizontalalignment='right')

    plt.show()


def plot_interactive(coords, record_solutions):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Epoch', 0, len(record_solutions) - 1, valinit=0, valfmt='%0.0f')

    def update(val):
        epoch = int(slider.val)
        plot_tour(ax, coords, record_solutions[epoch], title=f'TSP Tour at epoch {epoch}')
        fig.canvas.draw_idle()

    slider.on_changed(update)

    update(0)

    plt.show()


def run_simulation(filepath):
    coords = load_tsp(filepath)
    best_solution, best_fitness, record_best_fitness, record_solutions = simulated_annealing(coords)

    tsp_name = os.path.basename(filepath)
    plot_fitness(record_best_fitness, best_fitness, tsp_name)

    plot_interactive(coords, record_solutions)


tsp_filepath = 'TSP/lin105.tsp'
run_simulation(tsp_filepath)
