import numpy as np


class solution():
    def __init__(self, x):
        self.x = x
        self.y = None

    def fitness(self, problem):
        self.y = problem(self.x)

    def __call__(self):
        return self.x

    def __repr__(self) -> str:
        return f"x: {self.x}\ny: {self.y}"


class population():
    def __init__(self, size, lower_bound, upper_bound):
        self.size = size
        self.population = [solution(np.random.uniform(lower_bound, upper_bound))
                           for _ in range(size)]

    def __call__(self):
        return self.population

    def __getitem__(self, i):
        return self.population[i]

    def __setitem__(self, i, value):
        self.population[i] = value

    def __len__(self):
        return len(self.population)

    def __iter__(self):
        return iter(self.population)

    def __next__(self):
        return next(self.population)

    def __str__(self) -> str:
        return f"Population size: {self.size}"
    
    def __repr__(self) -> str:
        return f"Population size: {self.size}"