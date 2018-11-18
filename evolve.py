from eaters import *
class Sim():
    def __init__(self):
        pygame.init()
        size = 800, 800
        self.bg_color = pygame.Color('black')
        self.screen = pygame.display.set_mode(size)
        FOODS_COUNT = 10

        self.bugs = [Bug(self.screen)]
        self.foods = []

        food_locs = [(618, 771),(525, 287),(213, 643),(62, 573),(251, 527),(772, 62),(475, 715),(240, 438),(590, 185),(447, 314),(263, 618),(623, 200),(397, 169),(741, 135),(281, 90),(614, 483),(82, 787),(306, 55),(773, 366),(452, 276),(196, 633),(16, 760),(618, 240),(217, 555),(163, 386),(135, 667),(482, 779),(355, 173),(638, 301),(373, 519),(594, 581),(12, 184),(374, 553)]
        for i in range(0, FOODS_COUNT):
            self.foods.append(Food(self.screen))
            self.foods[i].set_loc(food_locs[i])

        self.sprites = self.foods + self.bugs
        self.t = 0
        self.kill = False

    def step(self, action, blit = False):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()
        self.screen.fill(self.bg_color)

        move = self.bugs[0].move(action)
        if not move:
            self.kill = True
            return False

        for bug in self.bugs:
            self.foods = bug.check_eat(self.foods)

        if blit:
            for sprite in self.sprites:
                sprite._blit()
            pygame.display.flip()
        self.t += TIME_CONST
        return True


    def get_descrete_action(self, action):
        output = []
        if action[0] > .5:
            output.append('u')
        if action[1] > .5:
            output.append('d')
        if action[2] > .5:
            output.append('r')
        if action[3] > .5:
            output.append('l')
        return output

    def get_scaled_state(self):
        bug = self.bugs[0]

        output = [0,0,0,0]
        bug_top = bug.rect.top
        bug_bottom = bug.rect.bottom
        bug_left = bug.rect.left
        bug_right = bug.rect.right
        distance = False

        for food in self.foods:
            food_top = food.rect.top
            food_bottom = food.rect.bottom
            food_left = food.rect.left
            food_right = food.rect.right

            #if  bug_top is in the same horizontal plane as the food:
            if (bug_top > food_top and bug_top < food_bottom) or (bug_bottom < food_bottom and bug_bottom > food_top):
                # if bug is to the left
                if bug_right < food_left:
                    distance_right = food_left - bug_right
                    normalized_right = self.normalize(distance_right, 0, bug.screen.get_width())
                else:
                    distance_right = (bug.screen.get_width() - bug_right)
                    normalized_right = self.normalize(distance_right, 0, bug.screen.get_width()) * -1
                # if bug is to the right
                if bug_left > food_right:
                    distance_left = bug_left - food_right
                    normalized_left = self.normalize(distance_left, 0, bug.screen.get_width())
                else:
                    distance_left = bug_left
                    normalized_left = self.normalize(distance_left, 0, bug.screen.get_width()) * -1
            else:
                # set the left and right distance to distance to edge
                # left
                distance_left = bug_left
                normalized_left = self.normalize(distance_left, 0, bug.screen.get_width()) * -1

                # right
                distance_right = (bug.screen.get_width() - bug_right)
                normalized_right = self.normalize(distance_right, 0, bug.screen.get_width()) * -1
            # if bug is in the same vertical plane as the food
            if (bug_left > food_left and bug_left < food_right) or (bug_right > food_left and  bug_right < food_left):
                # bug is below
                if bug_bottom > food_top:
                    distance_top = bug_bottom - food_top
                    normalized_top = self.normalize(distance_top, 0, bug.screen.get_height())
                else:
                    distance_top = bug_top
                    normalized_top = self.normalize(distance_top, 0, bug.screen.get_height()) * -1

                # bug is above
                if bug_top < food_bottom:
                    distance_bottom =  food_bottom - bug_top
                    normalized_bottom = self.normalize(distance_bottom, 0, bug.screen.get_height())
                else:
                    distance_bottom = (bug.screen.get_height() - bug_bottom)
                    normalized_bottom = self.normalize(distance_bottom, 0, bug.screen.get_height()) * -1

            else:
                # set the top and bottom distance to distance to edge
                # top
                distance_top = bug_top
                normalized_top = self.normalize(distance_top, 0, bug.screen.get_height()) * -1

                #bottom
                distance_bottom = (bug.screen.get_height() - bug_bottom)
                normalized_bottom = self.normalize(distance_bottom, 0, bug.screen.get_height()) * -1
        return [
            normalized_top, normalized_bottom, normalized_left, normalized_right
        ]

    def normalize(self, num, min_num, max_num):
        return (num - min_num) / (max_num - min_num)

    def get_fitness(self):
        return self.bugs[0].energy + (self.t*.01)


RUNS_PER_NET = 5
SIMULATION_SECONDS = 60.0
TIME_CONST = .01
def eval_genome(genome, config):
    global SIMULATION_SECONDS
    #net = neat.ctrnn.CTRNN.create(genome, config, TIME_CONST)
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []
    for run in range(RUNS_PER_NET):
        blit = (run == 0)

        sim = Sim()
        #net.reset()

        fitness = 0.0
        while sim.t < SIMULATION_SECONDS and sim.kill != True:
            inputs = sim.get_scaled_state()

            action = net.activate(inputs)

            success = sim.step(sim.get_descrete_action(action), blit)

            fitness = sim.get_fitness()
            fitnesses.append(fitness)
    SIMULATION_SECONDS += TIME_CONST
    return(sum(fitnesses)/len(fitnesses))

def eval_genomes(genomes, config):
    for gnome_id, genome in genomes:
        gnome.fitness = eval_gnome(gnome, config)

def run():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat.conf')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(3, eval_genome)
    winner = pop.run(pe.evaluate)

    # Save the winner.
    with open('winner-ctrnn', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)
    visualize.plot_stats(stats, ylog=True, view=True, filename="ctrnn-fitness.svg")
    visualize.plot_species(stats, view=True, filename="ctrnn-speciation.svg")
    node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}
    visualize.draw_net(config, winner, True, node_names=node_names)

    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-ctrnn.gv")
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-ctrnn-enabled.gv", show_disabled=False)
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-ctrnn-enabled-pruned.gv", show_disabled=False, prune_unused=True)
run()

