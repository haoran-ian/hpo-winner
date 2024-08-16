# from ConfigSpace import Configuration, ConfigurationSpace

# import numpy as np
# import opytimizer.optimizers
# import opytimizer.optimizers.swarm
# from smac import HyperparameterOptimizationFacade, Scenario
# from sklearn import datasets
# from sklearn.svm import SVC
# from sklearn.model_selection import cross_val_score

import ioh
import numpy as np
from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.spaces import SearchSpace
from opytimizer.optimizers.swarm import CS


class Algorithm_Evaluator():
    def __init__(self, optimizer):
        self.alg = optimizer

    def __call__(self, func, n_reps=5):
        def helper(x):
            return func(x.reshape(-1))
        space = SearchSpace(30, func.meta_data.n_variables,
                            func.bounds.lb, func.bounds.ub)
        optimizer = eval(f"{self.alg}()")
        function = Function(helper)
        for seed in range(n_reps):
            np.random.seed(int(seed))
            Opytimizer(space, optimizer, function).start(
                n_iterations=int((func.meta_data.n_variables * 10000) / 30))
            func.reset()


# Number of agents and decision variables
n_agents = 1
n_variables = 2
# Lower and upper bounds (has to be the same size as `n_variables`)
lower_bound = [-5. for _ in range(n_variables)]
upper_bound = [5. for _ in range(n_variables)]

prob = ioh.get_problem(fid=1, instance=1, dimension=n_variables,
                       problem_class=ioh.ProblemClass.BBOB)


def test_function(x):
    return prob(x)


# Creates the SearchSpace class
space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
# print(prob([1, 1]))
function = Function(test_function)
optimizer = CS()

opt = Opytimizer(space, optimizer, function)
opt.start()
