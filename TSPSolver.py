#!/usr/bin/python3
import copy
from random import randint
from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
import heapq
import itertools


def customSort(k):
    if k is None:
        return np.inf
    else:
        return k.cost


class TSPSolver:
    def __init__(self, gui_view):
        self.ncities = None
        self.cities = None
        self._scenario = None
        self.population_size = 100
        self.expanded_population = 200
        self.best_paths_size = 90

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    ###################################################################################################
    ''' <summary>
        This is the entry point for the default solver
        which just finds a valid random tour.  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of solution,
        time spent to find solution, number of permutations tried during search, the
        solution found, and three null values for fields not used for this
        algorithm</returns>
    '''

    # Finds a solution by random tour
    # Time Complexity: O(n!) - Worst Case / O(n) - Average Case
    # Space Complexity: O(n * n)
    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ###################################################################################################
    ''' <summary>
        This is the entry point for the greedy solver, which you must implement for
        the group project (but it is probably a good idea to just do it for the branch-and
        bound project as a way to get your feet wet).  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution,
        time spent to find best solution, total number of solutions found, the best
        solution found, and three null values for fields not used for this
        algorithm</returns>
    '''

    # Finds the best path by starting at each city and taking the smallest path out of the city
    # Input: time_allowance is how long in seconds the program can run, self which has n cities
    # Time Complexity: O(n * n * n) 
    # Space Complexity: O(n * n * n)
    # Return: The best greedy solution
    def greedy(self, time_allowance=60.0):
        results = {}
        self.cities = self._scenario.getCities()
        self.ncities = len(self.cities)
        count = 0
        # Find starting upper bound
        bssf = None
        paths = np.zeros((self.ncities, self.ncities))

        # Fills the paths matrix with the distace between all the cities
        # Time Complexity: O(n * n)
        # Space Complexity: O(n * n)
        for i in range(self.ncities):
            for j in range(self.ncities):
                if i == j:
                    paths[i][j] = math.inf
                else:
                    paths[i][j] = self.cities[i].costTo(self.cities[j])

        # Start time after initial variables are set up
        start_time = time.time()

        # Loops until all nodes have been tried for the greedy approach or time runs out
        # Time Complexity: O(n * n * n)
        # Space Complexity: O(n * n * n)
        i = 0
        while i < self.ncities and time.time() - start_time < time_allowance:
            solution = self.greedyHelper(i, 0, paths.copy(), [])
            if solution != None:
                count += 1
                if bssf == None or solution.cost <= bssf.cost:
                    bssf = solution
            i += 1

        end_time = time.time()
        results['cost'] = bssf.cost if bssf != None else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ###################################################################################################
    # Finds the best path by starting at each city and taking the smallest path possible out of the city, recursive
    # Input: Starting city, cost so far, paths, route so far
    # Time Complexity: O(n * (3n + 3)) - n times (recursive) and then 3n + 3 for each iteration
    # Space Complexity: O(n * n + n + 2)
    # Return: Greedy solution or None if path would be infinite
    def greedyHelper(self, start, cost, paths, route):
        # Base Case: If the tour is complete return
        if len(route) == self.ncities:
            return TSPSolution(route)

        # O(1)
        route.append(self.cities[start])

        # O(n)
        minimum_row = np.argmin(paths[start])

        # If the path is infinite then don't continue down the path and return
        if paths[start][minimum_row] == math.inf:
            return None
        cost += paths[start][minimum_row]
        paths[minimum_row][start] = math.inf

        # O(2n) Infinity out the visited node so that it can't be returned to
        paths[start] = [math.inf for i in range(self.ncities)]
        paths[:, minimum_row] = [
            math.inf for i in range(self.ncities)]

        # Time Complexity: O(n) to get through each
        # Space: O(n * n + n + 2)
        return self.greedyHelper(minimum_row, cost, paths, route)

    ###################################################################################################
    ''' <summary>
        This is the entry point for the branch-and-bound algorithm that you will implement
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution,
        time spent to find best solution, total number solutions found during search (does
        not include the initial BSSF), the best solution found, and three more ints:
        max queue size, total number of states created, and number of pruned states.</returns>
    '''

    # Finds the best path it can within the time limit before it returns the best solution found so far
    # Input: time_allowance is how long in seconds the program can run, self which has n cities
    # Time Complexity: O(n * n * n!) - Worst Case / O(n * n * b ^ n) - Average Case
    # Space Complexity: O(n * n * n!) - Worst Case / O(n * n * b ^ n) - Average Case
    # Return: The best solution so far, could be optimal
    def branchAndBound(self, time_allowance=60.0):
        results = {}
        self.cities = self._scenario.getCities()
        self.ncities = len(self.cities)
        self.count = 0
        self.max = 0
        self.total = 0
        self.pruned = 0
        self.queue = []
        heapq.heapify(self.queue)

        # Start time after initial variables are set up
        start_time = time.time()

        # Find starting upper bound with greedy (Max .1 seconds)
        # Time Complexity: O(n * n * n)
        # Space Complexity: O(n * n * n) 
        self.bssf = self.greedy(.1)['soln']

        # If no path found try 5 random and select the best one (Max .4 seconds)
        # Time Complexity: O(5 * n!) - Worst Case / O(5 * n) - Average Case
        # Space Complexity: O(5 * n * n)
        if self.bssf == None:
            potential_bssf = [self.defaultRandomTour(.08)['soln'] for i in range(5)]
            self.bssf = min(potential_bssf, key=lambda a: a.cost)

        paths = np.zeros((self.ncities, self.ncities))

        # Fills the paths matrix with the distace between all the cities
        # Time Complexity: O(n * n)
        # Space Complexity: O(n * n)
        for i in range(self.ncities):
            for j in range(self.ncities):
                if i == j:
                    paths[i][j] = math.inf
                else:
                    paths[i][j] = self.cities[i].costTo(self.cities[j])

        # Gets the initial reduced matrix and the lower bound
        # Time Complexity:  O(n * n)
        # Space Complexity: O(n * n)
        cost, paths = self.reduceRowCol(0, paths.copy())
        route = [self.cities[0]]
        self.total += 1

        # Finds the children of starting at city 0
        # Time Complexity O(n * n * n) - Worst Case / O(n * n * b) - Average Case
        # Space Complexity O(n * n * n) - Worst case / O(n * n * b) - Average Case
        self.findChildren(0, cost, paths.copy(), route.copy())

        # Loops until there is nothing left in the queue or there is no more time
        # Time Complexity: O(n * n * n!) - Worst Case / O(n * n * b ^ n) - Average Case
        # Space Complexity: O(n * n * n!) - Worst Case / O(n * n * b ^ n) - Average Case
        while len(self.queue) != 0 and time.time() - start_time < time_allowance:

            # Time complexity: O(mlogm)
            tspNode = heapq.heappop(self.queue)

            # Checks if the cost is less than the best so far cost since the best may have changed
            # then finds the children and pushes them to the queue
            # Time Complexity: O(n * n * n) - Worst Case / O(n * n * b) - Average Case
            # Space Comlexity: O(n * n * n) - Worst Case / O(n * n * b) - Average Case
            if tspNode.cost < self.bssf.cost:
                self.findChildren(tspNode.last, tspNode.cost,
                                  tspNode.paths, tspNode.route)
            else:
                self.pruned += 1

        end_time = time.time()
        results['cost'] = self.bssf.cost
        results['time'] = end_time - start_time
        results['count'] = self.count
        results['soln'] = self.bssf
        results['max'] = self.max
        results['total'] = self.total
        results['pruned'] = self.pruned
        return results

    ###################################################################################################
    # Finds the chilren of previous path that is not a complete tour
    # Input: start city which is the last in the route, cost so far, current paths, and the route so far
    # Time Complexity: O(n * (n * n + mlogm)) - Worst Case / O(b * (n * n + mlogm)) - Average Case
    # Space Comlexity: O(n * n * n) - Worst Case / O(n * n * b) - Average Case
    # Return: Void, however each valid child that has a current cost less that best so far is added to the queue
    def findChildren(self, start, cost, paths, route):
        for i in range(self.ncities):
            # If the path is not infinite find the next path
            # Time Complexity: O(n * n)
            # Space Complexity: O(n * n)
            if paths[start][i] != math.inf:
                self.total += 1
                tspNode = self.findChild(
                    start, i, cost, paths.copy(), route.copy())

                # If the cost is less than best so far push to queue or if tour complete, make bssf
                # Time Complexity: O(nlogn)
                # Space Complexity: 1 (Nothing new created)
                if tspNode.cost < self.bssf.cost:
                    if len(tspNode.route) != self.ncities:
                        heapq.heappush(self.queue, tspNode)
                        if len(self.queue) > self.max:
                            self.max = len(self.queue)
                    else:
                        self.bssf = TSPSolution(tspNode.route)
                        self.count += 1
                else:
                    self.pruned += 1

    ###################################################################################################
    # Finds a child of previous path that is not a complete tour
    # Input: start city, destination city, cost so far, current paths, and the route so far
    # Time Complexity: O(n * n + 2n + 3)
    # Spcae Complexity: O(n * n + n + 3)
    def findChild(self, start, dest, cost, paths, route):
        # Add destination to route and it's cost
        # Time Complexiy: O(2)
        # Space Complexiy: O(1)
        route.append(self.cities[dest])
        cost += paths[start][dest]

        # Infinity out the paths to the node
        # Time Complexiy: O(2n + 1)
        # Space Complexiy: O(1)
        paths[dest][start] = math.inf
        paths[start] = [math.inf for i in range(self.ncities)]
        paths[:, dest] = [
            math.inf for i in range(self.ncities)]

        # Reduce the matrix
        # Time Complexity: O(n * n)
        # Space Complexity: O(n * n)
        cost, paths = self.reduceRowCol(
            cost, paths.copy())
        return TSPNode(cost, paths, route, len(route), dest)

    ###################################################################################################
    # Reduced each row and column to have one zero in each
    # Input: Previous cost and paths
    # Time Complexity: O(n * n)
    # Space Complexity: O(n * n + 1)
    # Returns the new cost and reduced paths matrix
    def reduceRowCol(self, cost, paths):
        # Reduce Rows
        # Time Complexity: O(n * n)
        # Space Complexity: O(1)
        minimums_row = np.amin(paths, 1)
        for val in minimums_row:
            if val != math.inf:
                cost += val
        sub_paths_row = np.array(
            [[minimums_row[j] for i in range(len(paths))] for j in range(len(paths))])
        for i in range(len(paths)):
            for j in range(len(paths)):
                if (paths[i][j] == math.inf or sub_paths_row[i][j] == math.inf):
                    paths[i][j] = math.inf
                else:
                    paths[i][j] = paths[i][j] - sub_paths_row[i][j]

        # Reduce Columns
        # Time Complexity: O(n * n)
        # Space Complexity: O(1)
        minimums_col = np.amin(paths, 0)
        for val in minimums_col:
            if val != math.inf:
                cost += val
        sub_paths_col = np.array(
            [[minimums_col[i] for i in range(len(paths))] for j in range(len(paths))])
        for i in range(len(paths)):
            for j in range(len(paths)):
                if (paths[i][j] == math.inf or sub_paths_col[i][j] == math.inf):
                    paths[i][j] = math.inf
                else:
                    paths[i][j] = paths[i][j] - sub_paths_col[i][j]

        # O(n * n + 1)
        return cost, paths

    ###################################################################################################
    ''' <summary>
        This is the entry point for the algorithm you'll write for your group project.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number of solutions found during search, the 
        best solution found.  You may use the other three field however you like.
        algorithm</returns> 
    '''

    def fancy(self, time_allowance=60.0):
        cooled = 10000
        results = {}
        start_time = time.time()
        # Basic Genetic Algorithm
        #    Initial population
        population = self.generateInitialPopulation()
        while cooled != 0 and time.time() - start_time < time_allowance:
            #    Crossing over
            #    Mutation to introduce variation ?? Could be 'creative' with this
            for i in range(self.population_size):
                population[i + self.population_size] = self.mutatePath(population[i])
            #    Calculate fitness (not necessary, just cost of tour)
            #    Selecting best genes
            population = self.selectBestPaths(population)
            cooled -= 1
        # Also need to choose a cooling variable, where it stops after
        # a certain number of iterations or the timeout
        end_time = time.time()

        best_tour = population[0]
        results['cost'] = best_tour.cost
        results['time'] = end_time - start_time
        results['count'] = 0
        results['soln'] = best_tour
        results['max'] = 0
        results['total'] = 0
        results['pruned'] = 0
        return results

    def generateInitialPopulation(self):
        population = [None] * self.expanded_population  # Enough space to store the next generation
        population[0] = self.greedy()['soln']
        for i in range(1, self.population_size):
            population[i] = self.generateRandomTour()
        return population

    def generateRandomTour(self):
        self.cities = self._scenario.getCities()
        self.ncities = len(self.cities)
        perm = np.random.permutation(self.ncities)
        route = []
        for i in range(self.ncities):
            route.append(self.cities[perm[i]])
        tour = TSPSolution(route)
        return tour

    def selectBestPaths(self, old_population):
        new_population = [None] * self.expanded_population
        old_population.sort(key=customSort)
        population = copy.deepcopy(old_population)
        new_population[:self.best_paths_size] = population[:self.best_paths_size]
        for i in range(self.best_paths_size, self.population_size):
            new_population[i] = population[randint(self.best_paths_size, self.expanded_population - 1)]
        return new_population

    def crossOver(self, parent1, parent2):
        path1 = copy.deepcopy(parent1.route)
        path2 = copy.deepcopy(parent2.route)
        # do we care about the parent strings? then we can make a deep copy (will increase complexity)
        swap_index = self.ncities//4
        parent1_swap = path1[:swap_index]
        parent2_swap = path2[:swap_index]

        path1[:swap_index] = parent2_swap
        path2[:swap_index] = parent1_swap

        new_parent1 = TSPSolution(copy.deepcopy(path1))
        new_parent2 = TSPSolution(copy.deepcopy(path2))

        for i in range(self.ncities):
            temp = path1[0]
            del path1[0]
            if temp in path1:
                new_parent1.cost = math.inf
                break
        for i in range(self.ncities):
            temp = path2[0]
            del path2[0]
            if temp in path2:
                new_parent2.cost = math.inf
                break

        return new_parent1, new_parent2

    # do exploitative (local) search near the current solutions with mutation
    def mutatePath(self, tour):
        path = copy.deepcopy(tour.route)
        while True:
            r = randint(1, self.ncities-1)
            r1 = randint(1, self.ncities-1)
            if r1 != r:
                temp = path[r]
                path[r] = path[r1]
                path[r1] = temp
                break
        new_cost = TSPSolution(path)
        return new_cost
