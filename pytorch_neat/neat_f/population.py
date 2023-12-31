import logging
import random
import pickle
import numpy as np

#import neat.utils as utils
import neat_f.utils as utils
from neat_f.genotype.genome import Genome
from neat_f.species import Species
from neat_f.crossover import crossover
from neat_f.mutation import mutate

VERBOSE = True
STATS_FOLDER = 'stats/'

# CHANGE THIS FOR EVERY RUN
FOLDER_NAME = STATS_FOLDER + 'mod-hidden-test/'
logger = logging.getLogger(__name__)


class Population:
    
    __global_innovation_number = 0
    current_gen_innovation = []  # Can be reset after each generation according to paper

    def __init__(self, config):
        self.Config = config()
        self.population = self.set_initial_population()
        self.species = []

        for genome in self.population:
            self.speciate(genome, 0)

    def run(self):
        STATS_NAME = self.Config.FILE_NAME + "/"
        if VERBOSE:
                    print("STARTING A NEAT RUN")
        for generation in range(1, self.Config.NUMBER_OF_GENERATIONS):
            if VERBOSE:
                    print("GENERATION",generation)

            print("Computing Fitness...")
            # Get Fitness of Every Genome
            fitness = []
            point_visited = []
            for genome in self.population:
                fit,pv = self.Config.fitness_fn(genome)
                genome.fitness = max(0, fit)
                fitness_asInt = int(genome.fitness)
                fitness.append(fitness_asInt)
                point_visited.append(pv)

            print("Saving stats...")
            # Save Generation Stats
            gen_stats = {'fitness': fitness, 'point_visited': point_visited}
            with open(STATS_FOLDER + STATS_NAME + f'population_stats-gen{generation}.pkl', 'wb') as output:
                pickle.dump(gen_stats, output, 1)

            best_genome = utils.get_best_genome(self.population)

            # Save best genome
            with open('solutions/' + STATS_NAME + f'/bestGenome-gen{generation}.pkl', 'wb') as output:
                pickle.dump(best_genome, output, 1)

            print("Reproducing...")
            # Reproduce
            all_fitnesses = []
            remaining_species = []

            for species, is_stagnant in Species.stagnation(self.species, generation):
                if is_stagnant:
                    self.species.remove(species)
                else:
                    all_fitnesses.extend(g.fitness for g in species.members)
                    remaining_species.append(species)

            min_fitness = min(all_fitnesses)
            max_fitness = max(all_fitnesses)

            fit_range = max(1.0, (max_fitness-min_fitness))
            for species in remaining_species:
                # Set adjusted fitness
                avg_species_fitness = np.mean([g.fitness for g in species.members])
                species.adjusted_fitness = (avg_species_fitness - min_fitness) / fit_range

            adj_fitnesses = [s.adjusted_fitness for s in remaining_species]
            adj_fitness_sum = sum(adj_fitnesses)

            if VERBOSE:
                    print("Remaining Species:", len(remaining_species))

            # Get the number of offspring for each species
            new_population = []

            print("Selecting the new population...")
            for species in remaining_species:
                if species.adjusted_fitness > 0:
                    size = max(2, int((species.adjusted_fitness/adj_fitness_sum) * self.Config.POPULATION_SIZE))
                else:
                    size = 2

                # sort current members in order of descending fitness
                cur_members = species.members
                cur_members.sort(key=lambda g: g.fitness, reverse=True)
                species.members = []  # reset

                # save top individual in species
                new_population.append(cur_members[0])
                size -= 1

                # Only allow top x% to reproduce
                purge_index = int(self.Config.PERCENTAGE_TO_SAVE * len(cur_members))
                purge_index = max(2, purge_index)
                cur_members = cur_members[:purge_index]

                if VERBOSE:
                    print("The new resulting specie has size", size)

                for i in range(size):
                    parent_1 = random.choice(cur_members)
                    parent_2 = random.choice(cur_members)

                    child = crossover(parent_1, parent_2, self.Config)
                    mutate(child, self.Config)
                    new_population.append(child)

            # Set new population
            self.population = new_population

            if VERBOSE:
                print('Population STATS:')
                for genome in self.population:
                    print('num_nodes:', len(genome.node_genes))
            Population.current_gen_innovation = []

            # Speciate
            for genome in self.population:
                self.speciate(genome, generation)

            if best_genome.fitness >= self.Config.FITNESS_THRESHOLD:
                return best_genome, generation
            


            # Generation Stats
            if self.Config.VERBOSE:
                logger.info(f'Finished Generation {generation}')
                logger.info(f'Best Genome Fitness: {best_genome.fitness}')
                logger.info(f'Best Genome Length {len(best_genome.connection_genes)}\n')

            

        
        return best_genome, 'limit reached'
        return None, None

    def speciate(self, genome, generation):
        """
        Places Genome into proper species - index
        :param genome: Genome be speciated
        :param generation: Number of generation this speciation is occuring at
        :return: None
        """
        for species in self.species:
            if Species.species_distance(genome, species.model_genome) <= self.Config.SPECIATION_THRESHOLD:
                genome.species = species.id
                species.members.append(genome)
                return

        # Did not match any current species. Create a new one
        new_species = Species(len(self.species), genome, generation)
        genome.species = new_species.id
        new_species.members.append(genome)
        self.species.append(new_species)

    def assign_new_model_genomes(self, species):
        species_pop = self.get_genomes_in_species(species.id)
        species.model_genome = random.choice(species_pop)

    def get_genomes_in_species(self, species_id):
        return [g for g in self.population if g.species == species_id]

    def set_initial_population(self):
        pop = []
        for i in range(self.Config.POPULATION_SIZE):
            new_genome = Genome()
            inputs = []
            outputs = []
            HL1 = []
            HL2 = []
            HL3 = []
            HL4 = []
            bias = None

            print("Creating nodes...")
            # Create nodes
            for j in range(self.Config.NUM_INPUTS):
                n = new_genome.add_node_gene('input')
                #print('added node input')
                inputs.append(n)

            # Initialize hidden layers
            for j in range(self.Config.NUM_HIDDEN1):
                n = new_genome.add_node_gene('hidden')
                HL1.append(n)
                #print('added node hidden')

            for j in range(self.Config.NUM_HIDDEN2):
                n = new_genome.add_node_gene('hidden')
                HL2.append(n)

            for j in range(self.Config.NUM_HIDDEN3):
                n = new_genome.add_node_gene('hidden')
                HL3.append(n)

            for j in range(self.Config.NUM_HIDDEN4):
                n = new_genome.add_node_gene('hidden')
                HL4.append(n)
                
            # Create output nodes

            for j in range(self.Config.NUM_OUTPUTS):
                n = new_genome.add_node_gene('output')
                #print('added node output')
                outputs.append(n)

            if self.Config.USE_BIAS:
                bias = new_genome.add_node_gene('bias')

            print("Adding connections...")
            # Create connections
            for input in inputs:
                for hn1 in HL1:
                    new_genome.add_connection_gene(input.id, hn1.id)


            for hn1 in HL1:
                for hn2 in HL2:
                    new_genome.add_connection_gene(hn1.id, hn2.id)

            for hn2 in HL2:
                for hn3 in HL3:
                    new_genome.add_connection_gene(hn2.id, hn3.id)
            
            for hn3 in HL3:
                for hn4 in HL4:
                    new_genome.add_connection_gene(hn3.id, hn4.id)
            
            for hn4 in HL4:
                for output in outputs:
                    new_genome.add_connection_gene(hn4.id, output.id)

            # Add bias connection
            if bias is not None:
                for output in outputs:
                    new_genome.add_connection_gene(bias.id, output.id)

            pop.append(new_genome)
            print("Initial population created")
        return pop

    @staticmethod
    def get_new_innovation_num():
        # Ensures that innovation numbers are being counted correctly
        # This should be the only way to get a new innovation numbers
        ret = Population.__global_innovation_number
        Population.__global_innovation_number += 1
        return ret
