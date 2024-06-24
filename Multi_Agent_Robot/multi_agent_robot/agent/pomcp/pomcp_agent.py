from typing import Tuple, List

from .rock_sample_model import RockSampleModel
from .util.helper import rand_choice, randint, round
from .util.helper import elem_distribution, ucb
from .util.belief_tree import BeliefTree, ActionNode
from logger import Logger as log
import numpy as np
import time

from ..base import Agent
from ...env.history import History
from ...env.types import Action, State

MAX = np.inf

class UtilityFunction():

    @staticmethod
    def ucb1(c):
        def algorithm(action: ActionNode):
            return action.value + c * ucb(action.parent.visit_count, action.visit_count)
        return algorithm
    
    @staticmethod
    def mab_bv1(min_cost, c=1.0):
        def algorithm(action):
            if action.mean_cost == 0.0:
                return MAX
            ucb_value = ucb(action.parent.visit_count, action.visit_count)
            return action.mean_reward / action.mean_cost + c * ((1. + 1. / min_cost) * ucb_value) / (min_cost - ucb_value)
        return algorithm

    @staticmethod
    def sa_ucb(c0):
        def algorithm(action):
            if action.mean_cost == 0.0:
                return MAX
            return action.value + c0 * action.parent.budget * ucb(action.parent.visit_count, action.visit_count)
        return algorithm

UTILITY_FUNCTION_MAP = {
    'ucb1': UtilityFunction.ucb1,
    'mab_bv1': UtilityFunction.mab_bv1,
    'sa_ucb': UtilityFunction.sa_ucb
}

class POMCPAgent(Agent):
    def __init__(self, config_params: dict):
        self.tree = None
        self.simulation_time = None  # in seconds
        self.max_particles = None    # maximum number of particles can be supplied by hand for a belief node
        self.reinvigorated_particles_ratio = None  # ratio of max_particles to mutate
        self.max_simulation_depth = None  # ratio of max_particles to mutate
        self.budget = 0

        self.config_params = config_params
        self.add_configs(**config_params)
        self.model = RockSampleModel()

    def add_configs(self, name: str, simulation_time=0.5, timeout=30,
                    max_particles=80, reinvigorated_particles_ratio=0.1, utility_fn='ucb1', max_simulation_depth=5, no_particles = 1000, c=0.5):
        # acquaire utility function to choose the most desirable action to try
        self.name = name
        self.utility_fn = UTILITY_FUNCTION_MAP[utility_fn](c)
        self.timeout = timeout
        self.no_particles = no_particles

        # other configs
        self.simulation_time = simulation_time
        self.max_particles = max_particles
        self.reinvigorated_particles_ratio = reinvigorated_particles_ratio
        self.max_simulation_depth = max_simulation_depth

    def init_search_tree(self, state: State):
        # initialise belief search tree
        root_particles = self.model.gen_particles(state, n=self.max_particles)
        self.tree = BeliefTree(self.budget, root_particles)

    def compute_belief(self, state: State):
        base = [0.0] * self.model.num_states(state)
        particle_dist = elem_distribution(self.tree.root.B)
        for state, prob in particle_dist.items():
            base[self.model.states.index(state)] = round(prob, 6)
        return base

    def rollout(self, state:State, h, depth, max_depth, budget):
        """
        Perform randomized recursive rollout search starting from 'h' util the max depth has been achived
        :param state: starting state's index
        :param h: history sequence
        :param depth: current planning horizon
        :param max_depth: max planning horizon
        :return:
        """
        if depth > max_depth or budget <= 0:
            return 0

        ai = rand_choice(self.model.get_legal_actions(state))
        sj, oj, r, cost = self.model.simulate_action(state, ai)

        return r + self.model.discount_reward * self.rollout(sj, h + [ai, oj], depth + 1, max_depth, budget - cost)
        
    def simulate(self, state_hash, max_depth, depth=0, h=[], parent=None, budget=None):
        """
        Perform MCTS simulation on a POMCP belief search tree
        :param state_hash: starting state's index
        :return:
        """
        # Stop recursion once we are deep enough in our built tree
        if depth > max_depth:
            return 0

        obs_h = None if not h else h[-1]
        node_h = self.tree.find_or_create(h, name=obs_h or 'root', parent=parent,
                                          budget=budget, observation=obs_h)

        # ===== ROLLOUT =====
        # Initialize child nodes and return an approximate reward for this
        # history by rolling out until max depth
        if not node_h.children:
            # always reach this line when node_h was just now created
            for ai in self.model.get_legal_actions(State.from_hash(state_hash)):
                cost = self.model.cost_function(ai)
                # only adds affordable actions
                if budget - cost >= 0:
                    self.tree.add(h + [ai], name=ai, parent=node_h, action=ai, cost=cost)

            return self.rollout(state_hash, h, depth, max_depth, budget)

        # ===== SELECTION =====
        # Find the action that maximises the utility value
        np.random.shuffle(node_h.children)
        node_ha = sorted(node_h.children, key=self.utility_fn, reverse=True)[0]

        # ===== SIMULATION =====
        # Perform monte-carlo simulation of the state under the action

        sj, oj, reward, cost = self.model.simulate_action(State.from_hash(state_hash), node_ha.action)
        R = reward + self.model.discount_reward * self.simulate(sj, max_depth, depth + 1, h=h + [node_ha.action, oj],
                                                                parent=node_ha, budget=budget-cost)
        # ===== BACK-PROPAGATION =====
        # Update the belief node for h
        node_h.B += [state_hash]
        node_h.visit_count += 1

        # Update the action node for this action
        node_ha.update_stats(cost, reward)
        node_ha.visit_count += 1
        node_ha.value += (R - node_ha.value) / node_ha.visit_count

        return R

    def solve(self, state: State):
        """
        Solves for up to T steps
        """
        if not self.tree:
            self.init_search_tree(state)
        begin = time.time()
        n = 0
        while time.time() - begin < self.simulation_time:
            n += 1
            state = self.tree.root.sample_state()
            self.simulate(state, max_depth=self.max_simulation_depth, h=self.tree.root.history, budget=self.tree.root.budget)
        log.info('# Simulation = {}'.format(n))

    def get_action(self):
        """
        Choose the action maximises V
        'belief' is just a part of the function signature but not actually required here
        """
        root = self.tree.root
        action_vals = [(action.value, action.action) for action in root.children]
        return max(action_vals)[1]


    def update(self, state:  State, reward: float, last_action: Action, observation, history: History) -> Tuple[List[str], List[str]]:
        """
        Updates the belief tree given the environment feedback.
        extending the history, updating particle sets, etc
        """
        root = self.tree.root

        #####################
        # Find the new root #
        #####################
        new_root = root.get_child(last_action).get_child(observation)
        if new_root is None:
            log.warning("Warning: {} is not in the search tree".format(root.history + [last_action, observation]))
            # The step result randomly produced a different observation
            action_node = root.get_child(last_action)
            if action_node.children:
                # grab any of the beliefs extending from the belief node's action node (i.e, the nearest belief node)
                log.info('grabing a bearest belief node...')
                new_root = rand_choice(action_node.children)
            else:
                # or create the new belief node and rollout from there
                log.info('creating a new belief node')
                particles = self.model.gen_particles(state, n=self.max_particles)
                new_root = self.tree.add(history=action_node.history + [observation], name=observation, parent=action_node, observation=observation,
                                         particle=particles, budget=root.budget - action_node.cost)
        
        ##################
        # Fill Particles #
        ##################
        particle_slots = self.max_particles - len(new_root.B)
        if particle_slots > 0:
            # fill particles by Monte-Carlo using reject sampling
            particles = []
            while len(particles) < particle_slots:
                state_hash_i = root.sample_state()
                sj, oj, r, cost = self.model.simulate_action(State.from_hash(state_hash_i), last_action)

                if oj == observation:
                    particles.append(sj)
            new_root.B += particles

        #####################
        # Advance and Prune #
        #####################
        self.tree.prune(root, exclude=new_root)
        self.tree.root = new_root
        new_belief = self.compute_belief(state)
        return new_belief

    def act(self, state: State, history: History):
        self.solve(state)
        return self.get_action()

    def draw(self, beliefs):
        """
        Dummy
        """
        pass
