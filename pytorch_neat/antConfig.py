import torch
import gym
import numpy as np
import neat
from pyneat_repo.neat_f.phenotype.feed_forward import FeedForwardNet
from tqdm import tqdm 

def distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))


n_of_points = 10
points = np.random.rand(n_of_points, 3)

class ANTConfig:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VERBOSE = True

    NUM_INPUTS = n_of_points**2
    NUM_OUTPUTS = n_of_points
    USE_BIAS = True

    ACTIVATION = 'sigmoid'
    SCALE_ACTIVATION = 4.9

    FITNESS_THRESHOLD = 90.0

    POPULATION_SIZE = 10
    NUMBER_OF_GENERATIONS = 10
    SPECIATION_THRESHOLD = 3.0

    CONNECTION_MUTATION_RATE = 0.80
    CONNECTION_PERTURBATION_RATE = 0.90
    ADD_NODE_MUTATION_RATE = 0.03
    ADD_CONNECTION_MUTATION_RATE = 0.5

    CROSSOVER_REENABLE_CONNECTION_GENE_RATE = 0.25

    # Top percentage of species to be saved before mating
    PERCENTAGE_TO_SAVE = 0.80

    def fitness_fn(self, genome, points = points, n_ants=10, n_generations=10, alpha=1, beta=1, evaporation_rate=0.5, Q=1):
        # Initialize the network
        phenotype = FeedForwardNet(genome, self)
        
        

        # Do an ant colony run
        pheromone = np.ones((n_of_points, n_of_points))
        best_path = None
        best_path_length = np.inf
        best_score = 0
        best_individual_point_visited = 0
        for generation in tqdm(range(n_generations)):
            paths = []
            path_lengths = []
            scores = []

            for ant in range(n_ants):
                visited = [False]*n_of_points
                current_point = np.random.randint(n_of_points)
                path = [current_point]
                path_length = 0
                moves = 0 
                while moves < n_of_points:
                    probabilities = np.zeros(n_of_points)
                    visited[current_point] = True

                    ''' INPUTS FOR NEURAL NETWORK '''
                    ''' 
                        pheromone[current_point, unvisited_point]
                        distance(points[current_point], points[unvisited_point])
                        
                    '''
                    #points = np.random.rand(10, 3) # Generate 10 random 3D points 
                    #print(phenotype)

                    pheromone_aslist = torch.Tensor(pheromone).reshape(-1).to(self.DEVICE).unsqueeze(0)
                    network_input = pheromone_aslist
                    #print(network_input)
                    probabilities = phenotype(network_input)
                    #print('probs',probabilities)
                    ''' OUTPUTS FOR NEURAL NETWORK '''
                    ''' 
                        probabilities as np.zeros(len(unvisited))
                    '''
                    
                    #Softmax
                    probabilities = probabilities.squeeze(0).to('cpu').detach().numpy()
                    #print('probs',probabilities)
                    probabilities /= np.sum(probabilities)
                    
                    next_point = np.random.choice(range(n_of_points), p=probabilities)
                    path.append(next_point)
                    path_length += distance(points[current_point], points[next_point])
                    
                    current_point = next_point
                    moves += 1
                
                point_visited = visited.count(True) 
                paths.append(path)
                path_lengths.append(path_length)

                total_distance = distance(points[path[0]], points[path[-1]])
                for i in range(n_of_points-1):
                    total_distance += distance(points[path[i]], points[path[i+1]])
                score = (point_visited**2)*(1/total_distance)
                scores.append(score)
                
                if path_length < best_path_length:
                    best_path = path
                    best_path_length = path_length

                if score > best_score:
                    best_score = score
                    best_individual_point_visited = point_visited
            
            pheromone *= evaporation_rate
            
            # Update pheromone
            for path, path_length in zip(paths, path_lengths):
                for i in range(n_of_points-1):
                    pheromone[path[i], path[i+1]] += Q/path_length
                pheromone[path[-1], path[0]] += Q/path_length
        
        

        print('Best score:', best_score, best_individual_point_visited)
        #print('Best path:', best_path, '\n' 'Best path length:', best_path_length )
    
        return best_score

