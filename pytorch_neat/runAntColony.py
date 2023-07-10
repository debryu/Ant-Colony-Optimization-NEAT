import logging
import pickle
#import neat
import neat_f.population as pop
'''---------------------------------
        SET CONFIGURATION FILE
---------------------------------'''

# CHANGE THIS FOR EVERY RUN
#import antConfigPytorch as c
# OR
import antConfigMod as c


from neat_f.visualize import draw_net, draw_net_c
from tqdm import tqdm

SOLUTION_PATH = 'solutions/'

# CHANGE THIS FOR EVERY RUN
SOLUTION_FILE_NAME = 'mod-test-solution.pkl'

logger = logging.getLogger(__name__)

num_of_solutions = 0

avg_num_hidden_nodes = 0
min_hidden_nodes = 100000
max_hidden_nodes = 0
found_minimal_solution = 0

avg_num_generations = 0
min_num_generations = 100000


for i in tqdm(range(1)):
    neat = pop.Population(c.ANTConfig)
    solution, generation = neat.run()

    if generation == 'limit reached':
        generation = 5050505
        print('limit reached')
    if solution is not None:
        avg_num_generations = ((avg_num_generations * num_of_solutions) + generation) / (num_of_solutions + 1)
        min_num_generations = min(generation, min_num_generations)

        num_hidden_nodes = len([n for n in solution.node_genes if n.type == 'hidden'])
        avg_num_hidden_nodes = ((avg_num_hidden_nodes * num_of_solutions) + num_hidden_nodes) / (num_of_solutions + 1)
        min_hidden_nodes = min(num_hidden_nodes, min_hidden_nodes)
        max_hidden_nodes = max(num_hidden_nodes, max_hidden_nodes)
        if num_hidden_nodes == 1:
            found_minimal_solution += 1

        num_of_solutions += 1
        draw_net_c(c.ANTConfig, solution, view=True, filename='./images/solution-' + str(num_of_solutions), show_disabled=True)


if generation == 5050505:
    generation = 'limit reached'
# Save the solution
with open(SOLUTION_PATH + SOLUTION_FILE_NAME, 'wb') as output:
    saved_version = {'solution': solution, 'generation': generation}
    pickle.dump(solution, output, 1)


logger.info('Total Number of Solutions: ', num_of_solutions)
logger.info('Average Number of Hidden Nodes in a Solution', avg_num_hidden_nodes)
logger.info('Solution found on average in:', avg_num_generations, 'generations')
logger.info('Minimum number of hidden nodes:', min_hidden_nodes)
logger.info('Maximum number of hidden nodes:', max_hidden_nodes)
logger.info('Minimum number of generations:', min_num_generations)
logger.info('Found minimal solution:', found_minimal_solution, 'times')
