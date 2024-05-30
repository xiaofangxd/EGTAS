# -*- coding: utf-8 -*-
import geatpy as ea 


class soea_SEGA_templet(ea.SoeaAlgorithm):
    """
soea_SEGA_templet : class - Strengthen Elitist GA Algorithm

Algorithm description:
This algorithm class implements the strengthen elitist GA algorithm. The algorithm flow is as follows:
1) Initialize a population of N individuals according to the encoding rules.
2) Stop if the stopping condition is met, otherwise continue to execute.
3) Perform statistical analysis on the current population, such as recording its best individual, average fitness, etc.
4) Independently select N parents from the current population.
5) Independently perform crossover operations on these N parents.
6) Independently mutate these N crossover individuals.
7) Merge the parent population and the population obtained by crossover mutation to obtain a population of size 2N.
8) Select N individuals from the merged population according to the selection algorithm to obtain a new generation population.
9) Return to step 2.
This algorithm should set a larger crossover and mutation probability, otherwise there will be more and more duplicate individuals in the generated new generation population.
    
"""

    def __init__(self,
                 problem,
                 population,
                 MAXGEN=None,
                 MAXTIME=None,
                 MAXEVALS=None,
                 MAXSIZE=None,
                 logTras=None,
                 verbose=None,
                 outFunc=None,
                 drawing=None,
                 trappedValue=None,
                 maxTrappedCount=None,
                 dirName=None,
                 **kwargs):
        super().__init__(problem, population, MAXGEN, MAXTIME, MAXEVALS, MAXSIZE, logTras, verbose, outFunc, drawing, trappedValue, maxTrappedCount, dirName)
        if population.ChromNum != 1:
            raise RuntimeError('The population object passed in must be a single-chromosome population type.')
        self.name = 'SEGA'
        self.selFunc = 'tour' 
        if population.Encoding == 'P':
            self.recOper = ea.Xovpmx(XOVR=0.7) 
            self.mutOper = ea.Mutinv(Pm=0.5) 
        else:
            self.recOper = ea.Xovdp(XOVR=0.7)  # Generate two-point crossover operator object
            if population.Encoding == 'BG':
                self.mutOper = ea.Mutbin(Pm=None)  
            elif population.Encoding == 'RI':
                self.mutOper = ea.Mutbga(Pm=1 / self.problem.Dim, MutShrink=0.5, Gradient=20)  # Breeder GA mutation
                # self.mutOper = ea.Mutpolyn(Pm=1 / self.problem.Dim, DisI=20)  # Polynomial mutation
                
            else:
                raise RuntimeError('The encoding must be ''BG'', ''RI'' or ''P''.')

    def run(self, prophetPop=None):  # prophetPop is the prophet population (i.e. the population containing prior knowledge)
        # ==========================Initial configuration===========================
        population = self.population
        NIND = population.sizes
        self.initialization()  # Initialize some dynamic parameters of the algorithm class
        # ===========================Ready to evolve============================
        population.initChrom(NIND)  # Initialize the population chromosome matrix
        # Insert prior knowledge (note: the legitimacy of the prophet population prophetPop will not be checked here)
        if prophetPop is not None:
            population = (prophetPop + population)[:NIND]  # Insert the prophet population
        self.call_aimFunc(population)  # Calculate the objective function value of the population
        population.FitnV = ea.scaling(population.ObjV, population.CV, self.problem.maxormins)  # Calculate fitness
        # ===========================Start Evolving============================
        while not self.terminated(population):
            # Selection
            offspring = population[ea.selecting(self.selFunc, population.FitnV, NIND)]
            # Reproduction
            offspring.Chrom = self.recOper.do(offspring.Chrom)  # Crossover
            offspring.Chrom = self.mutOper.do(offspring.Encoding, offspring.Chrom, offspring.Field)  # Mutation
            self.call_aimFunc(offspring)  # Calculate fitness
            population = population + offspring  # Parent-offspring merge
            population.FitnV = ea.scaling(population.ObjV, population.CV, self.problem.maxormins)
            # Get a new population
            population = population[ea.selecting('dup', population.FitnV, NIND)]
        return self.finishing(population)
