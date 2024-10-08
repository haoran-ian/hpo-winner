#!/usr/bin/env python3
import sys
sys.path.append("..")
import shutil
from ioh import get_problem, ProblemClass, Experiment, logger
from coa.coa import COA
from ConfigSpace import Configuration, ConfigurationSpace
import numpy as np
from smac import HyperparameterOptimizationFacade, Scenario
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

dims = 2


def dat2runs(dat_path):
    f = open(dat_path, "r")
    lines = f.readlines()
    f.close()
    runs = []
    one_run = []
    for line in lines:
        if line[:11] == "evaluations":
            if one_run != []:
                runs += [np.array(one_run)]
            one_run = []
        else:
            elements = line[:-1].split(" ")
            one_run += [[float(elements[0]), float(elements[1])]]
    runs += [np.array(one_run)]
    return runs


def scoring_algorithm(dat_path):
    runs = dat2runs(dat_path)
    score = 0
    for run in runs:
        score += np.sum(run[:-1, 1] * (run[1:, 0] - run[:-1, 0]))
    return score / len(runs)


# score = scoring_algorithm(
#     "COA/COA_-tmp-1/data_f1_Sphere/IOHprofiler_f1_DIM2.dat")


def train(config: Configuration, seed: int=0) -> float:
    expCOA = Experiment(algorithm=COA(P_e=config["P_e"], max_pack_size=config["max_pack_size"]), fids=[1],
                        iids=[1], dims=[dims], reps=10,
                        problem_class=ProblemClass.BBOB,
                        output_directory="COA",
                        folder_name=f"COA_D{dims}",
                        algorithm_name=f"COA_D{dims}",
                        algorithm_info="",
                        store_positions=True,
                        merge_output=True,
                        zip_output=False,
                        remove_data=False)
    expCOA()
    # return np.random.uniform(0, 1)
    score = scoring_algorithm(f"COA/COA_D{dims}/data_f1_Sphere/IOHprofiler_f1_DIM{dims}.dat")
    shutil.rmtree(f"COA/COA_D{dims}/")
    return score


configspace = ConfigurationSpace({"P_e": (0.0001, 100.0),
                                  "max_pack_size": (1, 200)})
# Scenario object specifying the optimization environment
scenario = Scenario(configspace, output_directory="smac3_output/COA/",
                    deterministic=True, n_trials=200)

# Use SMAC to find the best configuration/hyperparameters
smac = HyperparameterOptimizationFacade(scenario, train)
incumbent = smac.optimize()


# expCOA = Experiment(algorithm=COA(P_e=0.1), fids=list(range(1, 25)),
#                     iids=[1], dims=[2], reps=10,
#                     problem_class=ProblemClass.BBOB,
#                     output_directory="COA",
#                     folder_name="COA_",
#                     algorithm_name="COA",
#                     algorithm_info="",
#                     store_positions=True,
#                     merge_output=True,
#                     zip_output=False,
#                     remove_data=False
#                     )
# expCOA()

# if __name__ == '__main__':
#     # seed = 42

#     dims = [2, 5, 10, 20]
#     reps = 5

#     # numpy.random.seed(seed)
#     # random.seed(seed)

#     expCOA = Experiment(algorithm=COA(P_e=0.1), fids=list(range(1, 25)), iids=[1],
#                         dims=dims, reps=reps,
#                         problem_class=ProblemClass.BBOB,
#                         output_directory="COA_data",
#                         folder_name="COA",
#                         algorithm_name="COA",
#                         algorithm_info="",
#                         store_positions=True,
#                         merge_output=True,
#                         zip_output=True,
#                         remove_data=True
#                         )
#     expCOA()
