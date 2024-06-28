from typing import Tuple, List

from .rock_sample_model import RockSampleModel
from .util.helper import rand_choice, randint, round
from .util.helper import elem_distribution, ucb
from .util.belief_tree import BeliefTree, ActionNode
from logger import Logger as log
import numpy as np
import time

from ..base import Agent
from ..oracle import OracleAgent
from ...env.history import History
from ...env.types import Action, State, SampleObservation, RobotActions

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

class POMCPAgent(OracleAgent):
    def __init__(self, config_params: dict):
        super().__init__(config_params)
        self.tree = None
        self.simulation_time = None  # in seconds
        self.max_particles = None    # maximum number of particles can be supplied by hand for a belief node
        self.reinvigorated_particles_ratio = None  # ratio of max_particles to mutate
        self.max_simulation_depth = None  # ratio of max_particles to mutate
        self.budget = 1

        self.config_params = config_params
        self.add_configs(**config_params)
        self.model = RockSampleModel()

    def add_configs(self, name: str, simulation_time=0.5,
                    max_particles=80, reinvigorated_particles_ratio=0.1, utility_fn='ucb1', max_simulation_depth=5, c=0.5):
        # acquaire utility function to choose the most desirable action to try
        self.name = name
        self.utility_fn = UTILITY_FUNCTION_MAP[utility_fn](c)

        # other configs
        self.simulation_time = simulation_time
        self.max_particles = max_particles
        self.reinvigorated_particles_ratio = reinvigorated_particles_ratio
        self.max_simulation_depth = max_simulation_depth

    def init_search_tree(self, state: State):
        # initialise belief search tree
        root_particles = self.model.gen_particles(state, n=self.max_particles)
        self.tree = BeliefTree(self.budget, root_particles)

    def update_belief(self, state: str, last_action: Action, observation: SampleObservation):
        if last_action.action_type is RobotActions.SAMPLE:
            rock_prob = self.rock_probs[last_action.rock_sample_loc]
            bad_rock_prob, good_rock_prob = self.get_bu_rock_probs(last_action.rock_sample_loc, rock_prob, observation, state)
            self.rock_probs[last_action.rock_sample_loc] = {SampleObservation.GOOD_ROCK: good_rock_prob,
                                                            SampleObservation.BAD_ROCK: bad_rock_prob}

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

        random_action = rand_choice(self.model.get_legal_actions(state))
        sj, oj, r, cost = self.model.simulate_action(state, random_action)
        return r + self.model.discount_reward * self.rollout(sj, h + [random_action, oj], depth + 1, max_depth, budget - cost)
        
    def simulate(self, state: State, max_depth, depth=0, cur_history=[], parent=None, budget=None):
        """
        Perform MCTS simulation on a POMCP belief search tree
        :param state: starting state's index
        :return:
        """
        # Stop recursion once we are deep enough in our built tree
        if depth > max_depth:
            return 0

        obs_h = None if not cur_history else cur_history[-1]
        belief_node = self.tree.find_or_create(cur_history, name=obs_h or 'root', parent=parent,
                                          budget=budget, observation=obs_h)

        # ===== ROLLOUT =====
        # Initialize child nodes and return an approximate reward for this
        # history by rolling out until max depth
        if not belief_node.children:
            # always reach this line when belief_node was just now created
            for ai in self.model.get_legal_actions(state):
                cost = self.model.cost_function(ai)
                # only adds affordable actions
                if budget - cost >= 0:
                    self.tree.add(cur_history + [ai], name=ai, parent=belief_node, action=ai, cost=cost)

            return self.rollout(state, cur_history, depth, max_depth, budget)

        # ===== SELECTION =====
        # Find the action that maximises the utility value
        np.random.shuffle(belief_node.children)
        action_node = sorted(belief_node.children, key=self.utility_fn, reverse=True)[0]

        # ===== SIMULATION =====
        # Perform monte-carlo simulation of the state under the action
        sj, oj, reward, cost = self.model.simulate_action(state, action_node.action)
        R = reward + self.model.discount_reward * self.simulate(sj, max_depth, depth + 1, cur_history=cur_history + [action_node.action, oj],
                                                                parent=action_node, budget=budget-cost)
        # ===== BACK-PROPAGATION =====
        # Update the belief node for h
        belief_node.add_particle(state)
        belief_node.visit_count += 1

        # Update the action node for this action
        action_node.update_stats(cost, reward)
        action_node.visit_count += 1
        action_node.value += (R - action_node.value) / action_node.visit_count

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
            self.simulate(state, max_depth=self.max_simulation_depth, cur_history=self.tree.root.history, budget=self.tree.root.budget)
        log.info('number of simulations done = {}'.format(n))
        return state

    def get_action(self)->ActionNode:
        """
        Choose the action maximises V
        'belief' is just a part of the function signature but not actually required here
        """
        root = self.tree.root
        action_vals = [(action.value, action) for action in root.children]
        return max(action_vals, key=lambda x:x[0])[1]


    def update(self, state:  State, reward: float, last_action: Action, observation: SampleObservation, history: History) -> Tuple[List[str], List[str]]:
        """
        Updates the belief tree given the environment feedback.
        extending the history, updating particle sets, etc
        """
        # oracle_action = self.oracle_act(state, last_action, observation, history)
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
        while len(new_root.belief_states) < self.max_particles:
            sampled_state = root.sample_state()
            sj, oj, r, cost = self.model.simulate_action(sampled_state, last_action)

            if oj == observation:
                new_root.add_particle(sj)

        #####################
        # Advance and Prune #
        #####################
        self.tree.prune(root, exclude=new_root)
        self.tree.root = new_root
        self.update_belief(state, last_action, observation)
        if last_action.action_type is RobotActions.SAMPLE:
            self.tree.root.update_particles_beliefs(state, last_action, observation, self.rock_probs[last_action.rock_sample_loc])

        return self.get_beliefs_as_db_repr(state, self.rock_probs), None, None #self.get_oracles_beliefs_as_db_repr(state), oracle_action

    def act(self, state: State, history: History):
        if all(state.collected_rocks()):
            return self.go_to_exit(state)
        simulated_state = self.solve(state)
        action = self.get_action()
        print(f"preforming action {action.action} assuming beliefs are {self.rock_probs}")
        return action.action

    def draw(self, beliefs):
        """
        Dummy
        """
        pass

# todo adjust the baysian update to update belief nodes only when a true observation is made
# sanity for the particles distibution
