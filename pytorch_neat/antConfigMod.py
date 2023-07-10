import torch
import pickle
import gym
import numpy as np
import neat
from neat_f.phenotype.feed_forward import FeedForwardNet
from tqdm import tqdm 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

IMPORT = True
SAVE_PHEROMONE = True # Use the same pheromone for all generations
# CHANGE THIS FOR EVERY RUN
PHEROMONE_FILE_NAME = 'mod-test'

file_path = 'pytorch_neat/pheromone/' + PHEROMONE_FILE_NAME + '.pkl'



def distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))


# Generate 10 random 3D points or import from file
if IMPORT:
    # Read the pickle file
    with open('pytorch_neat/points/v1.pkl', 'rb') as input:
        points = pickle.load(input)
    n_of_points = len(points)
else:
    n_of_points = 10
    points = np.random.rand(n_of_points, 3)


    

# Plot the points
for point in points:
    print(point)




class ANTConfig:
    

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DEVICE = torch.device("cpu")
    VERBOSE = True
    print("Using device: ", DEVICE)
    #NUM_INPUTS = 2*n_of_points**2 + n_of_points
    #Test
    #NUM_INPUTS = n_of_points
    #Test2
    NUM_INPUTS = 2*n_of_points 
    NUM_OUTPUTS = n_of_points 
    USE_BIAS = True

    ACTIVATION = 'sigmoid'
    SCALE_ACTIVATION = 4.9

    FITNESS_THRESHOLD = 160.0

    POPULATION_SIZE = 2
    NUMBER_OF_GENERATIONS = 100+1
    SPECIATION_THRESHOLD = 3.0

    CONNECTION_MUTATION_RATE = 0.80
    CONNECTION_PERTURBATION_RATE = 0.90
    ADD_NODE_MUTATION_RATE = 0.5
    ADD_CONNECTION_MUTATION_RATE = 0.5

    CROSSOVER_REENABLE_CONNECTION_GENE_RATE = 0.25

    # Top percentage of species to be saved before mating
    PERCENTAGE_TO_SAVE = 0.80

    def fitness_fn(self, genome, points = points, n_ants=8, n_generations=5, alpha=1, beta=1, evaporation_rate=0.5, Q=1):
        # Initialize the network
        phenotype = FeedForwardNet(genome, self)
        
        if os.path.exists(file_path):
            file_present = True
        else:
            file_present = False

        if SAVE_PHEROMONE:
            if file_present:
                with open(file_path, 'rb') as input:
                    saved_phero = pickle.load(input)
            else:
                print("File does not exist.")

        # Do an ant colony run
        if SAVE_PHEROMONE and file_present:
            pheromone = saved_phero
            #print('pheromone loaded')
        else:
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
                    probabilities = np.ones(n_of_points)
                    visited[current_point] = True

                    ''' INPUTS FOR NEURAL NETWORK '''
                    ''' 
                        pheromone[current_point, unvisited_point]
                        distance(points[current_point], points[unvisited_point])
                        
                    '''
                    #points = np.random.rand(10, 3) # Generate 10 random 3D points 
                    #print(phenotype)
                    
                    # Get the pheromone level of the current point
                    pheromone_current = pheromone[current_point]

                    #print('pheromone_current',pheromone_current)
                    # Convert pheromone to tensor
                    pheromone_asTensor = torch.Tensor(pheromone_current).reshape(-1).to(self.DEVICE).unsqueeze(0)

                    # Convert disances to tensor
                    distances = np.zeros((n_of_points, n_of_points))
                    for i in range(n_of_points):
                        for j in range(n_of_points):
                            distances[i,j] = distance(points[i], points[j])

                    # For now, we will use the distance matrix as input
                    # Since the matrix is symmetric, we can use only the upper triangular part
                    distances_asTensor = torch.Tensor(distances).reshape(-1).to(self.DEVICE).unsqueeze(0)


                    # Convert visited vector to tensor
                    # First convert to int
                    visited_asInt = [1 if i==True else 0 for i in visited]
                    visited_asTensor = torch.Tensor(visited_asInt).to(self.DEVICE).unsqueeze(0)

                    # Concatenate the two tensors
                    #network_input = torch.cat((pheromone_asTensor, distances_asTensor, visited_asTensor), dim=1)

                    #Test
                    #network_input = visited_asTensor
                    #Test2
                    network_input = torch.cat((pheromone_asTensor, visited_asTensor), dim=1)

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

                    if(np.sum(probabilities) == 0):
                        # If all probabilities are 0, we assign equal probabilities to all unvisited points
                        probabilities = np.ones(n_of_points)
                    
                    # Only uncomment to check for random results
                    #probabilities = np.ones(n_of_points)

                    probabilities /= np.sum(probabilities)
                    
                    #print('probs',probabilities)
                    next_point = np.random.choice(range(n_of_points), p=probabilities)
                    path.append(next_point)
                    path_length += distance(points[current_point], points[next_point])
                    
                    current_point = next_point
                    moves += 1
                
                point_visited = visited.count(True) 
                paths.append(path)
                path_lengths.append(path_length)

                
                total_distance = distance(points[path[0]], points[path[-1]]) + (10-point_visited) * 17.32
                for i in range(n_of_points-1):
                    total_distance += distance(points[path[i]], points[path[i+1]])

                # Score first attempt
                #score = (point_visited**2)*(1/total_distance)

                # Score second attempt
                #score = (point_visited/total_distance)

                # Score third attempt
                #score = (point_visited/total_distance #weight for choosing the shortest path
                #        + point_visited) #weight for choosing the path that visits the most points
                
                # Score fourth attempt
                #score = (40*point_visited/total_distance 
                #        + point_visited*10) 
                
                # Score fifth attempt   
                #score = (1/(total_distance-40)*500 
                #        + point_visited*10) 
                
                # Score sixth attempt   
                score = (-total_distance + 100
                        + point_visited*10) 
                # Score sixth attempt   
                score = -total_distance + 150
                        
                #if(point_visited == 10):
                #    score += 1000

                
                scores.append(score)
                
                if path_length < best_path_length:
                    best_path = path
                    best_path_length = path_length

                if score > best_score:
                    best_score = score
                    best_individual_point_visited = point_visited
                    best_total_distance = total_distance
            
            pheromone *= evaporation_rate

            # Update pheromone
            for path, path_length in zip(paths, path_lengths):
                if path_length == 0:
                    continue
                for i in range(n_of_points-1):
                    pheromone[path[i], path[i+1]] += Q/path_length
                pheromone[path[-1], path[0]] += Q/path_length

            # Normalize pheromone
            min_val = np.min(pheromone)
            max_val = np.max(pheromone)
            pheromone = (pheromone - min_val) * (10 - 0) / (max_val - min_val) + 0
        
        
        print('Best distance travelled:', best_total_distance)
        print('Best point visited:', best_individual_point_visited)
        print('Best score:', best_score)
        print('DPP:', best_total_distance/best_individual_point_visited)
        #print('Best path:', best_path, '\n' 'Best path length:', best_path_length )
    

        # Save the pheromone
        if SAVE_PHEROMONE:
            with open(file_path, 'wb') as output:
                pickle.dump(pheromone, output, 1)

        return best_score

