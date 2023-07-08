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
file_path = 'pytorch_neat/pheromone/P.pkl'
if os.path.exists(file_path):
    file_present = True
else:
    file_present = False


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

if SAVE_PHEROMONE:
    if file_present:
        with open(file_path, 'rb') as input:
            saved_phero = pickle.load(input)
    else:
        print("File does not exist.")
    

# Plot the points
for point in points:
    print(point)




class ANTConfig:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #DEVICE = torch.device("cpu")
    VERBOSE = True
    print("Using device: ", DEVICE)
    NUM_INPUTS = 2*n_of_points**2 + n_of_points
    #Test
    #NUM_INPUTS = n_of_points
    #Test2
    NUM_INPUTS = n_of_points**2
    NUM_OUTPUTS = n_of_points 
    USE_BIAS = True

    ACTIVATION = 'sigmoid'
    SCALE_ACTIVATION = 4.9

    FITNESS_THRESHOLD = 1000-400

    POPULATION_SIZE = 150
    NUMBER_OF_GENERATIONS = 10
    SPECIATION_THRESHOLD = 3.0

    CONNECTION_MUTATION_RATE = 0.80
    CONNECTION_PERTURBATION_RATE = 0.90
    ADD_NODE_MUTATION_RATE = 0.5
    ADD_CONNECTION_MUTATION_RATE = 0.5

    CROSSOVER_REENABLE_CONNECTION_GENE_RATE = 0.25

    # Top percentage of species to be saved before mating
    PERCENTAGE_TO_SAVE = 0.80

    def fitness_fn(self, genome, points = points, n_ants=10, n_generations=10, alpha=1, beta=1, evaporation_rate=0.5, Q=1):
        # Initialize the network
        phenotype = FeedForwardNet(genome, self)
        
        

        # Do an ant colony run
        if SAVE_PHEROMONE and file_present:
            pheromone = saved_phero
        else:
            pheromone = np.ones((n_of_points, n_of_points))

        best_score = 0

        '''
        COMPUTE ALL THE DISTANCES BETWEEN THE POINT
        '''

        points_asTensor = torch.tensor(points)

        id_of_pairs = torch.combinations(torch.tensor(range(n_of_points)), 2, with_replacement=False)
        #print(id_of_pairs.shape)
        #print(id_of_pairs)
        distances = torch.sqrt(torch.sum((points_asTensor[id_of_pairs[:, 0]] - points_asTensor[id_of_pairs[:, 1]])**2, dim=1))
        #print(distances)

        def distance_id(id1, id2):
            if(id1 > id2):
                id1, id2 = id2, id1
            
            return distances[n_of_points*id1 - id1*(id1+1)//2 + id2 - id1 - 1] 

        for generation in tqdm(range(n_generations)):
            
            pheromone_asTensor = torch.Tensor(pheromone).reshape(-1).to(self.DEVICE).unsqueeze(0)
            output_probabilities = phenotype(pheromone_asTensor).squeeze(0).softmax(dim=0).to(self.DEVICE)
            #print(output_probabilities)
            
            # Repeat the probability distribution for each ant
            probabilities = output_probabilities.unsqueeze(0).repeat(n_ants, 1)
        
            whole_run = torch.multinomial(probabilities, n_of_points, replacement=False)
            
            #print('whole run', whole_run)
            
            total_distances = torch.zeros(n_ants)
            for i in range(whole_run.shape[0]):
                #print('i', i)
                #print('whole run', whole_run[i, :])
                total_distances[i] += distance_id(whole_run[i, 0], whole_run[i, -1])
                for j in range(whole_run.shape[1]-1):
                    total_distances[i] += distance_id(whole_run[i, j], whole_run[i, j+1])
            
           
            #print('total_distances', total_distances)

                
            
            pheromone *= evaporation_rate
            
            # Update pheromone
            for ant in range(n_ants):
                if total_distances[ant] == 0:
                    continue
                for step in range(n_of_points-1):
                    pheromone[whole_run[ant, step], whole_run[ant, step+1]] += Q/total_distances[ant]
                pheromone[whole_run[ant, -1], whole_run[ant, 0]] += Q/total_distances[ant]

                
        best_total_distance = torch.min(total_distances)
        average_total_distance = torch.mean(total_distances)
        best_ant = torch.argmin(total_distances)
        best_path = whole_run[best_ant, :]
        best_score = 1000-best_total_distance*10
        average_score = 1000-average_total_distance*10


        print('Best distance travelled:', best_total_distance, '\nby ant with path:', best_path)
        print('Best score:', best_score, 'DPP:', best_total_distance/n_of_points)

        # Save the pheromone
        if SAVE_PHEROMONE:
            with open('pytorch_neat/pheromone/P.pkl', 'wb') as output:
                pickle.dump(pheromone, output, 1)

        #Test
        #return average_score
        return best_score

