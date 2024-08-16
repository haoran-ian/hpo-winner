import ioh
import numpy as np

from Solutions import population


def base_initialization(population):
    pass

def base_mutation(population):
    for solution in population:
        solution.x = np.random.uniform(-5, 5, 20)


solutions = population(10, [-5. for _ in range(20)], [5. for _ in range(20)])
print(solutions[2])
