import ast
import re

import numpy as np
import pandas as pd

from Multi_Agent_Robot.multi_agent_robot.env.types import RobotActions
from config import config

def parse_agent_rock_beliefs_list(beliefs_str: str):
    """Parse the `agent_rock_beliefs` string into a dictionary of (tuple) coordinates to float probabilities."""
    if beliefs_str == "[]" or not beliefs_str:
        return []  # Handle empty list case

    # Convert the string representation to a list of strings
    try:
        beliefs_list = ast.literal_eval(beliefs_str)
        beliefs_list = [float(entry.split(":")[1])for entry in beliefs_list]
        return beliefs_list
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing agent_rock_beliefs: {e}")
        return []


def parse_oracle_beliefs(beliefs_str: str)->list[list[float]]:
    """Parse the `agent_rock_beliefs` string into a dictionary of (tuple) coordinates to float probabilities."""
    if beliefs_str == "[]" or not beliefs_str:
        return []  # Handle empty list case

    # Convert the string representation to a list of strings
    try:
        beliefs_list = ast.literal_eval(beliefs_str)
        beliefs_list = [parse_agent_rock_beliefs_list(entry) for entry in beliefs_list]
        return beliefs_list
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing agent_rock_beliefs: {e}")
        return []

def enrich_real_rock_probs(game_name:str):
    rocks, rocks_reward = config.get_rocks(game_name=game_name)
    # Transform the rocks_reward (-15 -> 0, 15 -> 1)
    real_rock_probs = [1 if reward == 15 else 0 for reward in rocks_reward]
    return rocks,  real_rock_probs


def parse_action_values_with_enum(input_str):
    if input_str == '[]':
        return np.asarray([])
    # Number of actions based on the RobotActions enum
    num_actions = len(RobotActions)

    # Regex pattern to match each tuple (action, value)
    pattern = r"\(\s*(\w+)\s*,\s*([\d.-]+)\)"

    # Find all outer lists in the string
    outer_lists = re.findall(r"\[\[.*?\]\]", input_str)

    # Initialize an empty list to hold the parsed data
    parsed_data = []

    # Process each outer list separately
    for outer_list in outer_lists:
        # Find all inner lists within the outer list
        inner_lists = re.findall(r"\[.*?\]", outer_list)
        parsed_outer = []

        for inner_list in inner_lists:
            # Initialize a vector with NaNs for the actions
            action_values = np.full(num_actions, np.nan)

            # Find all action-value pairs in the inner list
            for action, value in re.findall(pattern, inner_list):
                try:
                    # Map action to its enum index and store the value
                    action_index = RobotActions[action].value
                    action_values[action_index] = float(value)
                except KeyError:
                    # If the action is not in the enum, ignore it
                    pass

            # Append the parsed action values for the inner list
            parsed_outer.append(action_values)

        # Append the parsed data for this outer list
        parsed_data.append(parsed_outer)

    # Convert to a 3D NumPy array
    return np.array(parsed_data)


def parse_values_ordered(input_str,  order = None)->list[float]:
    # Extract tuples and values from the input string
    matches = re.findall(r"\((\d+), (\d+)\):([\d.]+)", input_str)

    # Convert matches into a dictionary for easy lookup
    values_dict = {(int(x), int(y)): float(value) for x, y, value in matches}

    # Arrange values based on the order in `rocks`
    ordered_values = [values_dict[tuple(rock)] for rock in order]
    return ordered_values


import pandas as pd


def split_by_repeating_steps(df, step_col='step'):
    """
    Splits the DataFrame into a list of DataFrames based on dynamic repeating sequences in the step column.

    Args:
        df (pd.DataFrame): The DataFrame containing the simulation data with a repeating step column.
        step_col (str): The name of the column containing the step data.

    Returns:
        list[pd.DataFrame]: A list of DataFrames, each containing one detected cycle of steps.
    """
    steps = df[step_col].tolist()  # Extract the list of steps
    split_dfs = []  # List to store the resulting DataFrames
    current_cycle = []  # To accumulate rows in the current cycle
    last_step = steps[0]  # Initialize with the first step in the list

    # Iterate through the DataFrame rows to detect cycles
    for index, step in enumerate(steps):
        if index > 0 and step <= last_step:  # A new cycle likely starts
            # If the current cycle is not empty, append it as a DataFrame to split_dfs
            split_dfs.append(df.iloc[index - len(current_cycle):index].reset_index(drop=True))
            current_cycle = []  # Reset the cycle accumulator

        # Add the current row to the cycle
        current_cycle.append(step)
        last_step = step

    # Append the final cycle if there are any remaining rows
    if current_cycle:
        split_dfs.append(df.iloc[len(steps) - len(current_cycle):].reset_index(drop=True))

    return split_dfs
