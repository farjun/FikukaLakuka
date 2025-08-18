from abc import abstractmethod
from copy import copy
from typing import Union, List, Dict

import numpy as np

from config import config
from Multi_Agent_Robot.multi_agent_robot.env.types import SampleObservation, Action, State, RobotActions
from . import mult_on_axis
from .helper import rand_choice, round


class Node(object):
    def __init__(self, nid, name, h, tree_ptr: "BeliefTree", parent=None, value=0, visit_count=0):
        self.history = h
        self.visit_count = visit_count
        self.id = nid
        self.value = value
        self.tree_ptr = tree_ptr
        self.name = name
        self.parent = parent
        self.children: Union[List[ActionNode], List[BeliefNode]] = []

    @abstractmethod
    def add_child(self, node):
        """
         To be implemented.
        """

    @abstractmethod
    def get_child(self, *args):
        """
         To be implemented.
        """


class BeliefNode(Node):
    """
    Represents a node that holds the belief distribution given its history sequence in a belief tree.
    It also holds the received observation after which the belief is updated accordingly
    """

    def __init__(self, nid, name, h, obs_index, tree_ptr: "BeliefTree", parent=None, value=0, visit_count=0, budget=float('inf'), particles=None):
        Node.__init__(self, nid, name, h, tree_ptr, parent, value, visit_count)
        self.observation = obs_index
        self.budget = budget
        self.belief_states: list[State] = particles
        self.action_map = {}
        self.belief_states_probs: np.array = []


    def add_child(self, node):
        self.children.append(node)
        self.action_map[node.action] = node

    def get_child(self, action: Action):
        return self.action_map.get(action)

    def get_q_values(self, mean=False, max_q=False):
        if not self.children:
            return 0

        if mean:
            return sum(child.value for child in self.children) / len(self.children)

        if max_q:
            return max(child.value for child in self.children)

    def sample_state(self, rock_probs, size=None) -> State:
        belief_states_probs = self.get_belief_states_probs(rock_probs, self.belief_states)
        return rand_choice(np.asarray(self.belief_states), p=belief_states_probs, size=size).deep_copy()

    @staticmethod
    def calc_belief_states_rock_mask(belief_states: list[State]):
        new_rock_mask = []
        for p in belief_states:
            new_rock_mask.append([r.reward for r in p.rocks])
        new_rock_mask = np.asarray(new_rock_mask)
        return (new_rock_mask / (State.ASSUMED_ROCK_REWARD * 2)) + 0.5


    def update_belief_states_probs(self, rock_probs: dict[tuple, dict[SampleObservation, float]]):
        """
        Updates the belief distribution given the observation and action
        """
        self.belief_states_probs = self.get_belief_states_probs(rock_probs, self.belief_states)


    def get_belief_states_probs(self, rock_probs: dict[tuple, dict[SampleObservation, float]], belief_states: list[State]):
        rock_probs_arr = np.asarray(
            [[rock_probs[r.loc][SampleObservation.GOOD_ROCK], rock_probs[r.loc][SampleObservation.BAD_ROCK]] for r in belief_states[0].rocks])
        bad_rock_mask = 1 - self.tree_ptr.BELIEF_STATES_ROCK_MASK
        good_rocks_res = mult_on_axis(self.tree_ptr.BELIEF_STATES_ROCK_MASK, rock_probs_arr.T[0], axis=1)
        bad_rocks_res = mult_on_axis(bad_rock_mask, rock_probs_arr.T[1], axis=1)
        states_probs = np.prod(good_rocks_res + bad_rocks_res, axis=1)
        prob_sum = np.sum(states_probs)
        if prob_sum == 0:
            states_probs = np.ones(len(states_probs))
            prob_sum = len(states_probs)
        return states_probs / prob_sum

    def __repr__(self):
        return 'BeliefNode({}, visits = {})'.format(self.observation, self.visit_count)

    def __copy__(self):
        bn = BeliefNode(self.id, self.name, self.history, self.observation, self.parent, self.value, self.visit_count,
                        self.budget)
        bn.belief_states = self.belief_states.copy()
        return bn

    def __eq__(self, other):
        return self.id == other.id


class ActionNode(Node):
    """
    represents the node associated with an POMDP action
    """

    def __init__(self, nid, name, h, action_index:Action, cost, tree_ptr: "BeliefTree", parent=None, value=0, visit_count=0):
        Node.__init__(self, nid, name, h, tree_ptr, parent, value, visit_count)
        self.direct_reward = 0.0
        self.mean_action_reward = 0.0
        self.mean_cost = 0.0
        self.mean_future_reward = 0.0
        self.max_future_reward = -np.inf
        self.cost = cost
        self.action = action_index
        self.obs_map = {}

    def update_stats(self, cost, cur_reward, future_reward):
        self.mean_cost = (self.mean_cost * self.visit_count + cost) / (self.visit_count + 1)
        self.mean_action_reward = (self.mean_action_reward * self.visit_count + cur_reward) / (self.visit_count + 1)
        self.mean_future_reward = (self.mean_future_reward * self.visit_count + future_reward) / (self.visit_count + 1)
        self.max_future_reward = max(self.max_future_reward, future_reward)
        self.direct_reward = cur_reward
        self.visit_count += 1
        self.value = self.mean_action_reward + self.mean_future_reward + 0.2*self.max_future_reward


    @property
    def grandchildren(self):
        return [child.children for child in self.children if child.children]

    def add_child(self, node):
        self.children.append(node)
        self.obs_map[node.observation] = node

    def get_child(self, observation: SampleObservation):
        return self.obs_map.get(observation, None)

    def __repr__(self)->str:
        return self.action.action_type.name

    def to_ui_name(self)->str:
        action_name = str(self.action.action_type.name)
        if self.action.rock_sample_loc is not None:
            action_name += f" {self.action.rock_sample_loc}"
        return action_name

    def __copy__(self):
        an = ActionNode(self.id, self.name, self.history, self.action, self.cost, self.tree_ptr, self.parent, self.value,
                        self.visit_count)
        an.mean_action_reward = self.mean_action_reward
        an.mean_cost = self.mean_cost
        an.obs_map = self.obs_map.copy()
        return an


class BeliefTree:
    """
    The belief tree decipted in Silver's POMCP paper.
    """

    def __init__(self, total_budget, root_particles):
        """
        :param root_particles: particles sampled from the prior belief distribution; used as initial root's particle set
        """
        self.counter = 0
        self.nodes = {}
        self.root = self.add(history=[], name='root', particle=root_particles, budget=total_budget)
        # setup attribute for all nodes
        self.BELIEF_STATES = root_particles
        self.BELIEF_STATES_ROCK_MASK = BeliefNode.calc_belief_states_rock_mask(root_particles)

    def to_db_str(self, max_depth=4)->str:
        res_str = str(self.root) + "\n"
        return self._to_db_str(self.root, 0, res_str, max_depth)


    def _to_db_str(self, root, depth, res_str, max_depth=4)->str:
        if not root.children:
            # the leaf
            return res_str

        for node in root.children:
            if node.visit_count > 0 and depth < max_depth:
                res_str += '|  ' * depth + str(node) + "\n"
                res_str = self._to_db_str(node, depth + 1, res_str)
        return res_str


    def __pretty_print__(self, root, depth, skip_unvisited=False):
        if not root.children:
            # the leaf
            return

        for node in root.children:
            if (skip_unvisited and node.visit_count > 0) or not skip_unvisited:
                print('|  ' * depth + str(node))
                self.__pretty_print__(node, depth + 1, skip_unvisited=skip_unvisited)

    def add(self, history, name, parent=None, action=None, observation=None, particle=None, budget=None, cost=None):
        """
        Creates and adds a new belief node or action node to the belief search tree

        :param history: history sequence
        :param parent: either ActionNode or BeliefNode
        :param action: action name
        :param observation: observation name
        :param particle: new node's particle set
        :param budget: remaining budget of a belief nodde
        :param cost: action cost of an action node
        :return:
        """
        history = history[:]

        # instantiate node
        if action is not None:
            node = ActionNode(self.counter, name, history, tree_ptr=self, parent=parent, action_index=action, cost=cost)
        else:
            node = BeliefNode(self.counter, name, history, tree_ptr=self, parent=parent, obs_index=observation, budget=budget, particles = particle)

        # add the node to belief tree
        self.nodes[node.id] = node
        self.counter += 1

        if parent is not None:
            parent.add_child(node)
        return node

    def find_or_create(self, h, **kwargs) -> Union[BeliefNode, ActionNode]:
        """
        Search for the node corrresponds to given history, otherwise create one using given params
        """
        curr = self.root
        h_len, root_history_len = len(h), len(self.root.history)

        for step in range(root_history_len, h_len):
            curr = curr.get_child(h[step])
            if curr is None:
                node = self.add(h, **kwargs)
                return node
        return curr

    def prune(self, node, exclude=None):
        """
        Removes the entire subtree subscribed to 'node' with exceptions.
        :param node: root of the subtree to be removed
        :param exclude: exception component
        :return:
        """
        for child in node.children:
            if exclude and exclude.id != child.id:
                self.prune(child, exclude)

        self.nodes[node.id] = None
        del self.nodes[node.id]

    def prune_siblings(self, node):
        siblings = [child for child in node.parent.children if child.id != node.id]
        for sb in siblings:
            self.prune(sb)

    def pretty_print(self, skip_unvisited=False):
        """
         pretty prints tree's structure
        """
        print(str(self.root))
        self.__pretty_print__(self.root, depth=1, skip_unvisited=skip_unvisited)

    def copy(self) -> 'BeliefTree':
        bt = BeliefTree(0, self.root.belief_states)
        bt.root = copy(self.root)
        for node_id, node in self.nodes.items():
            if node is not None:
                bt.nodes[node_id] = copy(node)
            else:
                bt.nodes[node_id] = None
        return bt
