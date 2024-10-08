import numpy as np

#source code of WOA is publicly available and has been used for WOA implementation as a starting point
#code can be found at https://seyedalimirjalili.com/woa
#Source code for thesis Tijn van Son (s2589842) Universiteit Leiden

class WhaleOptimization():
    def __init__(self, problem, nsols, b, a, a_step, maximize=False, budget=None):
        self.problem = problem
        self.budget = 20000 
        self._constraints = list(zip(problem.bounds.lb, problem.bounds.ub)) 
        self._sols = self._init_solutions(nsols) 
        self._b = b
        self._a = a
        self._a_step = a_step
        self._maximize = maximize
        self._best_solutions = []
        self.total_budget = self.budget
        
    def get_solutions(self):
        return self._sols
                  
    
    def optimize(self):
        while self.budget > 0:
            ranked_sol = self._rank_solutions()
            best_sol = ranked_sol[0]  #this is the solution with the best fitness from the previous iteration
            new_sols = [best_sol] #the best solution gets carried over to the new solutions

            for s in ranked_sol[1:]:  
                if self.budget <= 0: 
                    break
                self.budget -= 1  
                new_s = None
                
                if np.random.uniform(0.0, 1.0) > 0.5:
                    A = self._compute_A()
                    norm_A = np.linalg.norm(A)  #normalize A

                    if norm_A < 1.0:
                        new_s = self._encircle(s, best_sol, A)

                    else:
                        random_sol = self._sols[np.random.randint(self._sols.shape[0])]
                        new_s = self._search(s, random_sol, A)

                else:
                    new_s = self._attack(s, best_sol)
                    
                new_s = self._constrain_solution(new_s) #solutions with positions outside of the constraints get constrained to the border of these constraints
                new_sols.append(new_s)
                if self.budget <= 0: 
                    break
            self._sols = np.stack(new_sols)
            
            if(self._a > 0): #a is decreasing from 2 to 0
                self._a -= self._a_step
           

    def _init_solutions(self, nsols): #we start initializing solutions with obtaining random solutions within the constraints
        sols = []
        for c in self._constraints:
            sols.append(np.random.uniform(c[0], c[1], size=nsols))
                                                                            
        sols = np.stack(sols, axis=-1)
        return sols
    
    #this function is the constrained method used for this algorithm. It sets the positions that are outside the bounds to the border of this bound
    def _constrain_solution(self, sol):
        """ensure solutions are valid wrt to constraints"""
        constrain_s = []
        for c, s in zip(self._constraints, sol):
            if c[0] > s:
                s = c[0]
            elif c[1] < s:
                s = c[1]
            constrain_s.append(s)
        return constrain_s

    #This function sorts all solutions based on fitness with the solution ranked from best to worst
    def _rank_solutions(self): 
        fitnesses = [self.problem(s) for s in self._sols]
        sol_fitness = list(zip(fitnesses, self._sols))
        ranked_sol = list(sorted(sol_fitness, key=lambda x:x[0], reverse=self._maximize))
        self._best_solutions.append(ranked_sol[0])
        return [ s[1] for s in ranked_sol] 

    def print_best_solutions(self):
        print('generation best solution history')
        print('([fitness], [solution])')
        for s in self._best_solutions:
            print(s)
        print('\n')
        print('best solution')
        print('([fitness], [solution])')
        print(sorted(self._best_solutions, key=lambda x:x[0], reverse=self._maximize)[0])

    #This function computes the coefficient vector A
    def _compute_A(self):
        r = np.random.uniform(0.0, 1.0, size=self._sols[0].shape)
        return (2.0*np.multiply(self._a, r))-self._a

    #This function computes the coefficient vector C
    def _compute_C(self):
        return 2.0*np.random.uniform(0.0, 1.0, size=self._sols[0].shape)
                     
    #This function represents the encircle phase                                        
    def _encircle(self, sol, best_sol, A):
        D = self._encircle_D(sol, best_sol)
        return best_sol - np.multiply(A, D)
               
    #This function calculates the distance vector D for the encircling phase. 
    #The whale will update its position towards the best current search agent                                               
    def _encircle_D(self, sol, best_sol):
        C = self._compute_C()
        return np.multiply(C, best_sol) - sol 

    #this function represents the search for prey phase
    def _search(self, sol, rand_sol, A):
        D = self._search_D(sol, rand_sol)
        return rand_sol - np.multiply(A, D)

    #This function calculates the distance vector for updating the position towards a random chosen search agent
    def _search_D(self, sol, rand_sol):
        C = self._compute_C() 
        return np.multiply(C, rand_sol) - sol     

    #This function represent the second approach of the bubble-net attacking method phase
    def _attack(self, sol, best_sol):
        D = best_sol - sol #This represents D'
        L = np.random.uniform(-1.0, 1.0, size=len(sol)) #this represents parameter l with a interval of [-1, 1]
        spiral_component = np.multiply(np.exp(self._b * L) * np.cos(2 * np.pi * L), D)
        return best_sol + spiral_component