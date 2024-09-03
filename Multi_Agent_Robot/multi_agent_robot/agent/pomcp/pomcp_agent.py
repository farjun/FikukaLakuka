from typing import Tuple, List, Dict

from .rock_sample_model import RockSampleModel
from .util import BeliefNode
from .util.helper import rand_choice, randint, round, draw_arg
from .util.helper import elem_distribution, ucb
from .util.belief_tree import BeliefTree, ActionNode
from logger import Logger as log
import numpy as np
import time

from ..base import Agent
from ..oracle import OracleAgent
from ...env.history import History
from ...env.multi_agent_robot import MultiAgentRobotEnv
from ...env.types import Action, State, SampleObservation, RobotActions

MAX = np.inf

class UtilityFunction:

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

    @staticmethod
    def softmax(action_vals:list[tuple]):
        """Compute softmax values for each sets of scores in x."""
        values, actions = list(zip(*action_vals))
        values = np.asarray(values)
        e_x = np.exp(values - np.max(values))
        return np.random.choice(actions, p=e_x / e_x.sum(axis=0))

    @staticmethod
    def max(action_vals):
        """Compute softmax values for each sets of scores in x."""
        return max(action_vals, key=lambda x:x[0])[1]

SAMPLE_ACTION_FUNCTION_MAP = {
    'ucb1': UtilityFunction.ucb1,
    'mab_bv1': UtilityFunction.mab_bv1,
    'sa_ucb': UtilityFunction.sa_ucb
}

SELECT_ACTION_FUNCTION_MAP = {
    'softmax': UtilityFunction.softmax,
    'max': UtilityFunction.max,
}

class POMCPAgent(OracleAgent):
    def __init__(self, config_params: dict):
        super().__init__(config_params)
        self.tree = None
        self.simulation_time = None  # in seconds
        self.simulation_iters = None
        self.max_particles = None    # maximum number of particles can be supplied by hand for a belief node
        self.reinvigorated_particles_ratio = None  # ratio of max_particles to mutate
        self.max_simulation_depth = None  # ratio of max_particles to mutate
        self.budget = 1

        self.config_params = config_params
        self.add_configs(**config_params)
        self.model = RockSampleModel()

    def gen_particles(self, state: State, n):
        states, prob = state.get_all_possible_belief_states(self.rock_probs)
        prob = np.array(prob)
        return list(np.random.choice(states, p=prob / prob.sum(), size=n))

    def add_configs(self, name: str, simulation_time=0.5,
                    max_particles=80, reinvigorated_particles_ratio=0.1, utility_fn='ucb1', max_simulation_depth=5, c=0.5, action_selection_strategy= 'max', simulation_iters=None):
        # acquaire utility function to choose the most desirable action to try
        self.name = name
        self.utility_fn = SAMPLE_ACTION_FUNCTION_MAP[utility_fn](c)
        self.action_selection_strategy = SELECT_ACTION_FUNCTION_MAP[action_selection_strategy]

        # other configs
        self.simulation_time = simulation_time
        self.simulation_iters = simulation_iters

        self.max_particles = max_particles
        self.reinvigorated_particles_ratio = reinvigorated_particles_ratio
        self.max_simulation_depth = max_simulation_depth

    def init_search_tree(self, state: State):
        # initialise belief search tree
        root_particles = self.gen_particles(state, n=self.max_particles)
        self.tree = BeliefTree(self.budget, root_particles)

    def update_belief(self, state: str, last_action: Action, observation: SampleObservation):
        if last_action.action_type is RobotActions.SAMPLE:
            rock_prob = self.rock_probs[last_action.rock_sample_loc]
            bad_rock_prob, good_rock_prob = self.get_bu_rock_probs(last_action.rock_sample_loc, rock_prob, observation, state)
            self.rock_probs[last_action.rock_sample_loc] = {SampleObservation.GOOD_ROCK: good_rock_prob,
                                                            SampleObservation.BAD_ROCK: bad_rock_prob}

    def rollout(self, state:State, cur_history, belief_node : BeliefNode, depth : int, budget):
        """
        Perform randomized recursive rollout search starting from 'h' util the max depth has been achived
        :param state: starting state's index
        :param h: history sequence
        :param depth: current planning horizon
        :return:
        """
        if depth > self.max_simulation_depth or budget <= 0:
            return 0

        random_action = rand_choice(Action.all_actions(state, cur_history, include_buy_information=False))
        sj, oj, r, cost = self.simulate_action(state, random_action)
        cur_history += [random_action]
        # action_node = self.tree.add(cur_history, name=random_action, parent=belief_node, action=random_action, cost=cost)
        cur_history += [oj]
        # belief_node = self.tree.add(cur_history, name=oj, observation=oj, parent=action_node, cost=cost, budget=belief_node.budget - action_node.cost)
        return r + self.model.discount_reward * self.rollout(sj, cur_history, belief_node, depth + 1, budget - cost)
        
    def simulate(self, state: State, depth=0, cur_history=[], parent=None, budget=None):
        """
        Perform MCTS simulation on a POMCP belief search tree
        :param state: starting state's index
        :return:
        """
        # Stop recursion once we are deep enough in our built tree
        if depth > self.max_simulation_depth:
            return 0

        obs_h = None if not cur_history else cur_history[-1]
        belief_node = self.tree.find_or_create(cur_history, name=obs_h or 'root', parent=parent,  budget=budget, observation=obs_h)

        # ===== ROLLOUT =====
        # Initialize child nodes and return an approximate reward for this
        # history by rolling out until max depth
        if not belief_node.children:
            # always reach this line when belief_node was just now created
            for ai in Action.all_actions(state, cur_history, include_buy_information=self.in_a_simulation):
                cost = self.model.cost_function(ai)
                # only adds affordable actions
                if budget - cost >= 0:
                    self.tree.add(cur_history + [ai], name=ai, parent=belief_node, action=ai, cost=cost)

            return self.rollout(state, cur_history.copy(), belief_node, depth, budget)

        # ===== SELECTION =====
        # Find the action that maximises the utility value
        np.random.shuffle(belief_node.children)
        action_node = sorted(belief_node.children, key=self.utility_fn, reverse=True)[0]

        # ===== SIMULATION =====
        # Perform monte-carlo simulation of the state under the action
        sj, oj, reward, cost = self.simulate_action(state, action_node.action)
        R = reward + self.model.discount_reward * self.simulate(sj, depth + 1, cur_history=cur_history.copy() + [action_node.action, oj],
                                                                parent=action_node, budget=budget-cost)
        # ===== BACK-PROPAGATION =====
        # Update the belief node for h
        belief_node.add_particle(state)
        belief_node.visit_count += 1

        # Update the action node for this action
        action_node.update_stats(cost, reward)
        action_node.visit_count += 1
        action_node.value = max(R, action_node.value)

        return R

    def solve(self, state: State):
        """
        Solves for up to T steps
        """
        if not self.tree:
            self.init_search_tree(state)

        begin = time.time()
        n = 0
        while ((self.simulation_time and time.time() - begin < self.simulation_time) or (self.simulation_iters and  n < self.simulation_iters)):
            n += 1
            state = self.tree.root.sample_state(self.rock_probs)
            self.simulate(state, cur_history=self.tree.root.history, budget=self.tree.root.budget)
        if not self.in_a_simulation:
            log.info('number of simulations done = {}'.format(n))
        return state

    def get_action(self)->ActionNode:
        """
        Choose the action maximises V
        'belief' is just a part of the function signature but not actually required here
        """
        action_vals = self.root_action_q_values()
        return self.action_selection_strategy(action_vals)

    def root_action_q_values(self):
        return [(action.value, action) for action in self.tree.root.children]

    def update(self, state:  State, reward: float, last_action: Action, observation: SampleObservation, history: History) -> Tuple[List[str], List[str]]:
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
                particles = self.gen_particles(state, n=self.max_particles)
                new_root = self.tree.add(history=action_node.history + [observation], name=observation, parent=action_node, observation=observation,
                                         particle=particles, budget=root.budget - action_node.cost)
        
        ##################
        # Fill Particles #
        ##################
        particles_to_add = list()
        while len(new_root.belief_states) + len(particles_to_add) < self.max_particles:
            sampled_state = root.sample_state(self.rock_probs)
            sj, oj, r, cost = self.simulate_action(sampled_state, last_action)
            if oj == observation:
                particles_to_add.append(sj)

        if particles_to_add:
            new_root.add_particle(particles_to_add)

        #####################
        # Advance and Prune #
        #####################
        self.tree.prune(root, exclude=new_root)
        self.tree.root = new_root
        self.update_belief(state, last_action, observation)
        self.tree.root.update_particles_beliefs(self.rock_probs)

        oracle_action = self.oracle_act(state, last_action, observation, history)
        return oracle_action

    def act(self, state: State, history: History)->Action:
        if all(state.collected_rocks()):
            return self.go_to_exit(state)
        self.solve(state)
        action = self.get_action()
        if not self.in_a_simulation:
            print(f"Robot preforming action {action.action}")
        return action.action

    def enter_inner_simulation_mode(self, beliefs: Dict[tuple, Dict[SampleObservation, float]], backup_data = False):
        super().enter_inner_simulation_mode(beliefs)
        if backup_data:
            self._backup_tree = self.tree
            self.tree = None

    def exit_inner_simulation_mode(self, restore_data = False):
        super().exit_inner_simulation_mode()
        if restore_data:
            self.tree = self._backup_tree

    def get_history_data(self, state, history)->dict:
        return {
            "agent_beliefs": self.get_beliefs_as_db_repr(state, self.rock_probs),
            "oracle_beliefs" : self.get_oracles_beliefs_as_db_repr(state),
            "agent_tree": self.tree.to_db_str(max_depth=4),
            "agent_belief_states": str(self.tree.root.belief_states_rock_mask),
            "agent_belief_states_probs": str(self.tree.root.belief_states_probs),
        }

    def simulate_action(self, state: State, ai: Action = None):
        """
        Query the resultant new state, observation and rewards, if action ai is taken from state si

        si: current state
        ai: action taken at the current state
        return: next state, observation and reward
        """
        observation, reward, done, state = MultiAgentRobotEnv.transotion_state(state.deep_copy(), action=ai)
        return state, observation, reward, 0


# todo adjust the baysian update to update belief nodes only when a true observation is made
# sanity for the particles distibution
