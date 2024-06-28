from typing import Union, List, Dict

from Multi_Agent_Robot.multi_agent_robot.env.types import SampleObservation, Action, State, RobotActions, RockTile
from .helper import rand_choice, round
from abc import abstractmethod

class Node(object):
    def __init__(self, nid, name, h, parent=None, value=0, visit_count=0):
        self.history = h
        self.value = value
        self.visit_count = visit_count
        self.id = nid
        self.name = name
        self.parent = parent
        self.children : Union[List[ActionNode], List[BeliefNode]]= []

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
    def __init__(self, nid, name, h, obs_index, parent=None, value=0, visit_count=0, budget=float('inf')):
        Node.__init__(self, nid, name, h, parent, value, visit_count)
        self.observation = obs_index
        self.budget = budget
        self.belief_states: List[State] = []
        self.action_map = {}
        self.belief_states_agg_probs: List[int] = []

    @property
    def belief_states_probs(self)->List[float]:
        obs_count_sum = sum(self.belief_states_agg_probs)
        if obs_count_sum == 0:
            return None
        return [1 / obs_count_sum * count for count in self.belief_states_agg_probs]

    def add_child(self, node):
        self.children.append(node)
        self.action_map[node.action] = node

    def get_child(self, action):
        return self.action_map.get(action)

    def sample_state(self):
        return rand_choice(self.belief_states, p=self.belief_states_probs)

    def add_particle(self, particle: List[State]):
        if type(particle) is list:
            self.belief_states.extend(particle)
            self.belief_states_agg_probs.extend([0] * len(particle))
        else:
            self.belief_states.append(particle)
            self.belief_states_agg_probs.append(0)



    def update_particles_beliefs(self, state: State, action: Action, observation: SampleObservation, rock_probs: Dict[SampleObservation, float]):
        """
        Updates the belief distribution given the observation and action
        """
        sampled_rock_index = state.rocks.index(state.rocks_map[action.rock_sample_loc])
        for i, state_hash in enumerate(self.belief_states):
            state = state_hash

            if int(state.rocks[sampled_rock_index].reward > 0) == observation.value:
                # good observation on a belief state set this rock to a good rock
                self.belief_states_agg_probs[i] = rock_probs[SampleObservation.GOOD_ROCK]
            else:
                self.belief_states_agg_probs[i] = rock_probs[SampleObservation.BAD_ROCK]

    def __repr__(self):
        return 'BeliefNode({}, visits = {}, cur_belief_probs={})'.format(self.observation, self.visit_count, self.belief_states_probs)


class ActionNode(Node):
    """
    represents the node associated with an POMDP action
    """
    def __init__(self, nid, name, h, action_index, cost, parent=None, value=0, visit_count=0):
        Node.__init__(self, nid, name, h, parent, value, visit_count)
        self.mean_reward = 0.0
        self.mean_cost = 0.0
        self.cost = cost
        self.action = action_index
        self.obs_map = {}

    def update_stats(self, cost, reward):
        self.mean_cost = (self.mean_cost * self.visit_count + cost) / (self.visit_count + 1)
        self.mean_reward = (self.mean_reward * self.visit_count + reward) / (self.visit_count + 1)

    def add_child(self, node):
        self.children.append(node)
        self.obs_map[node.observation] = node

    def get_child(self, observation):
        return self.obs_map.get(observation, None)

    def __repr__(self):
        return 'Action({}, visits = {}, value = {})'.format(self.action, self.visit_count, round(self.value, 6))


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

    def __pretty_print__(self, root, depth, skip_unvisited = False):
        if not root.children:
            # the leaf
            return

        for node in root.children:
            if (skip_unvisited and node.visit_count > 0) or not skip_unvisited:
                print('|  ' * depth + str(node))
                self.__pretty_print__(node, depth + 1, skip_unvisited=skip_unvisited)

    def add(self, history, name, parent=None, action=None, observation=None,
            particle=None, budget=None, cost=None):
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
            node = ActionNode(self.counter, name, history, parent=parent, action_index=action, cost=cost)
        else:
            node = BeliefNode(self.counter, name, history, parent=parent, obs_index=observation, budget=budget)

        if particle is not None:
            node.add_particle(particle)

        # add the node to belief tree
        self.nodes[node.id] = node
        self.counter += 1

        # register node as parent's child
        if parent is not None:
            parent.add_child(node)
        return node

    def find_or_create(self, h, **kwargs)->Union[BeliefNode, ActionNode]:
        """
        Search for the node corrresponds to given history, otherwise create one using given params
        """
        curr = self.root
        h_len, root_history_len = len(h), len(self.root.history)

        for step in range(root_history_len, h_len):
            curr = curr.get_child(h[step])
            if curr is None:
                return self.add(h, **kwargs)
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

    def pretty_print(self, skip_unvisited = False):
        """
         pretty prints tree's structure
        """
        print(self.root)
        self.__pretty_print__(self.root, depth=1, skip_unvisited=skip_unvisited)
