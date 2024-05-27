import os

import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time


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



def calculate_distance_matrix(coords):
    num_cities = len(coords)
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            dist_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])
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


def rotate_subsection(tour):
    new_tour = tour.copy()
    i, j = sorted(random.sample(range(len(tour)), 2))
    section = new_tour[i:j + 1]
    rotated_section = section[1:] + section[:1]
    new_tour[i:j + 1] = rotated_section
    return new_tour



def simulated_annealing(coords, initial_temperature=100, cooling=0.9, computing_time=1):
    dist_matrix = calculate_distance_matrix(coords)
    num_cities = len(coords)

    current_solution = list(range(num_cities))
    random.shuffle(current_solution)
    best_solution = current_solution
    best_fitness = objective_function(best_solution, dist_matrix)
    current_temperature = initial_temperature
    record_best_fitness = []

    start = time.time()
    while time.time() - start < computing_time:
        new_solution = random.choice([swap_cities, reverse_subsection, insert_city, rotate_subsection])(
            current_solution)
        current_fitness = objective_function(current_solution, dist_matrix)
        new_fitness = objective_function(new_solution, dist_matrix)

        if new_fitness < best_fitness or random.random() < math.exp(
                (current_fitness - new_fitness) / current_temperature):
            current_solution = new_solution
            if new_fitness < best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness

        record_best_fitness.append(best_fitness)
        current_temperature *= cooling

    return best_solution, best_fitness, record_best_fitness

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

def plot_tour(coords, tour, title='TSP Tour'):
    plt.figure(figsize=(10, 5))
    plt.plot(coords[tour][:, 0], coords[tour][:, 1], 'o-', label='Tour')
    plt.plot([coords[tour[-1]][0], coords[tour[0]][0]], [coords[tour[-1]][1], coords[tour[0]][1]],
             'o-')  # Close the loop
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()


def run_simulation(filepath):
    coords = load_tsp(filepath)
    best_solution, best_fitness, record_best_fitness = simulated_annealing(coords)

    plt.plot(record_best_fitness)
    plt.title('Best Fitness Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('Total Distance')
    plt.show()

    plot_tour(coords, best_solution, title='Best TSP Tour')
    tsp_name = os.path.basename(filepath)
    diff = get_diff_result(tsp_name, best_fitness)
    print(f"Difference from Optimal: {diff}")



tsp_filepath = 'TSP/lin105.tsp'
run_simulation(tsp_filepath)
