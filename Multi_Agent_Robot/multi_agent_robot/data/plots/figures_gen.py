from plotly.graph_objs import Figure
import plotly.express as px

import numpy as np
import pandas as pd

from Multi_Agent_Robot.multi_agent_robot.data.process.utils import enrich_real_rock_probs, split_by_repeating_steps
import plotly.graph_objects as go
from .oracle_analysis import OracleAnalyzer

class PlotsGenerator:

    def __init__(self):
        self.figures = list()

    def add_figure(self, fig: Figure):
        if fig is not None:
            self.figures.append(fig)

    def get_figures(self, game_data: dict)->list[Figure]:
        """
        Generate Plotly figures based on the env_history data.
        """
        # Create a figure for agent reward as a function of step
        env_df = game_data['env_data']
        game_name = game_data['game_name']
        simulations_data = game_data['simulations_data']

        # Add existing plots
        self.add_figure(self.plot_reward(game_name, env_df, simulations_data))
        self.add_figure(self.plot_agent_rock_belief_loss(game_name, env_df, simulations_data))
        self.add_figure(self.plot_prob_of_agent_belief_in_oracle_beliefs(game_name, env_df, simulations_data))
        self.add_figure(self.plot_simulation_rewards_per_step(game_name, env_df, simulations_data))
        
        # Add comprehensive Oracle analysis plots
        try:
            oracle_analyzer = OracleAnalyzer(env_df, simulations_data)
            oracle_figures = oracle_analyzer.get_all_figures()
            for fig in oracle_figures:
                self.add_figure(fig)
            
            # Add Oracle insights summary
            insights = oracle_analyzer.get_decision_insights()
            self.add_figure(self.create_oracle_insights_summary(insights))
            
        except Exception as e:
            print(f"Warning: Could not generate Oracle analysis plots: {e}")

        return self.figures

    def plot_prob_of_agent_belief_in_oracle_beliefs(self, game_name, env_df, simulations_data)->Figure:
        # Oracle Belief Probability Plot
        # Process data to get matching probabilities
        matching_probs = []
        for _, row in env_df.iterrows():
            # Round values in 'oracle_beliefs_parsed' and 'agent_rock_beliefs_list' to 3 decimals
            oracle_beliefs_rounded = [[round(val, 3) for val in belief] for belief in row['oracle_beliefs_list']]
            agent_belief_rounded = [round(val, 3) for val in row['agent_rock_beliefs_list']]

            # Find if agent's true belief matches any of the oracle's beliefs
            if agent_belief_rounded in oracle_beliefs_rounded:
                match_index = oracle_beliefs_rounded.index(agent_belief_rounded)
                matching_prob = row['oracle_beliefs_dist'][match_index]
            else:
                matching_prob = 0  # No match found

            matching_probs.append(matching_prob)
        # Add the new column to the DataFrame for plotting
        env_df['matching_probability'] = matching_probs
        # Create the matching probability plot
        fig_matching_prob = px.line(
            env_df,
            x='step',
            y='matching_probability',
            title="Probability of Agent's True Belief in Oracle Distributions"
        )
        fig_matching_prob.update_layout(xaxis_title="Step", yaxis_title="Matching Probability")
        return fig_matching_prob

    def plot_agent_rock_belief_loss(self, game_name, env_df, simulations_data)->Figure:
        fig = go.Figure()
        rocks, real_probs = enrich_real_rock_probs(game_name.rsplit("_", maxsplit=1)[0])
        numpy_rows = env_df['agent_rock_probs_dist_from_real_probs'].to_numpy()
        if len(numpy_rows) > 0 and len(numpy_rows[0]) > 0:
            rock_prob_dist = np.stack(numpy_rows)

            for i, rock_cord in enumerate(rocks):
                # Add a trace for each rock coordinate with its corresponding color
                fig.add_trace(go.Scatter(
                    x=env_df['step'],
                    y=rock_prob_dist[:, i],
                    mode='lines+markers',
                    name=f'Rock {rock_cord}',
                    line=dict(width=2),
                ))
        fig.update_layout(title="Agent Rock belief dist from truth")

        return fig

    def plot_reward(self, game_name, env_df, simulations_data):
        fig = px.line(env_df, x='step', y='cumulative_reward', title="Agent Reward vs Step")
        fig.update_layout(xaxis_title="Step", yaxis_title="Reward")
        return fig

    def plot_simulation_rewards_per_step(self, game_name, env_df, simulations_data):
        """
        Plot rewards from multiple runs of oracle simulations with and without help for each step in env_df.

        Args:
            game_name (str): The name of the game.
            env_df (pd.DataFrame): DataFrame containing environment data for each step.
            simulations_data (dict): Dictionary containing simulation data by step.

        Returns:
            go.Figure: A Plotly figure showing the rewards from multiple simulation runs per step.
        """
        # Initialize a Plotly figure
        fig = go.Figure()

        # Track the default visibility of each trace
        default_visibility = []

        # Iterate over each step in env_df to plot rewards for each simulation type
        for step in env_df['step'].unique():
            # Identify the simulation tables for the current step
            help_simulation_name = f'oracle_simulation_help_step_{step}_history'
            no_help_simulation_name = f'oracle_simulation_no_help_step_{step}_history'

            # Extract and split reward data for each run from the "help" simulation
            if help_simulation_name in simulations_data:
                help_runs = split_by_repeating_steps(simulations_data[help_simulation_name])
                for run_idx, run_df in enumerate(help_runs, start=1):
                    run_df['cumulative_reward'] = run_df['reward'].cumsum()
                    cur_simulated_changed_rock = run_df['cur_simulated_changed_rock'].loc[0]
                    unchanged_rock_beliefs = run_df['unchanged_rock_beliefs_list'].loc[0]
                    start_rock_beliefs = run_df['agent_rock_beliefs'].loc[0]

                    fig.add_trace(go.Scatter(
                        x=run_df['step'],
                        y=run_df['cumulative_reward'],
                        mode='lines+markers',
                        name=f'Simulation Run (step {step}) - Help {cur_simulated_changed_rock} - belief: {start_rock_beliefs}',
                        line=dict(dash='solid'),
                        visible=True if step == 1 else False  # Only Step 1 is visible by default
                    ))
                    default_visibility.append(step == 1)  # Track visibility for initial state

            # Extract and split reward data for each run from the "no help" simulation
            if no_help_simulation_name in simulations_data:
                no_help_runs = split_by_repeating_steps(simulations_data[no_help_simulation_name])
                for run_idx, run_df in enumerate(no_help_runs, start=1):
                    run_df['cumulative_reward'] = run_df['reward'].cumsum()
                    unchanged_rock_beliefs = run_df['unchanged_rock_beliefs_list'].loc[0]
                    start_rock_beliefs = run_df['agent_rock_beliefs'].loc[0]

                    fig.add_trace(go.Scatter(
                        x=run_df['step'],
                        y=run_df['cumulative_reward'],
                        mode='lines+markers',
                        name=f'Simulation Run (step {step}) - No Help - belief {unchanged_rock_beliefs}',
                        line=dict(dash='dot'),
                        visible=True if step == 1 else False  # Only Step 1 is visible by default
                    ))
                    default_visibility.append(step == 1)  # Track visibility for initial state

        # Update layout to add filtering by simulation type and run
        fig.update_layout(
            title=f"Oracle's Simulation runs Reward per Step",
            xaxis_title="Step",
            yaxis_title="Reward",
            legend_title="Simulation Run: ",
            hovermode="x unified"  # Makes it easier to compare values at each step
        )

        # Add dropdown buttons to filter by each step
        buttons = [
            dict(label="Show All Step 1 Runs",
                 method="update",
                 args=[{"visible": [step == 1 for step in env_df['step'] for _ in
                                    range(len(simulations_data) // len(env_df['step'].unique()))]}]),
        ]

        # Add buttons for each step's runs
        for step in env_df['step'].unique():
            step_visibility = [trace.name.startswith(f"Simulation Run (step {step})") for trace in fig.data]
            buttons.append(
                dict(label=f"Show Step {step} Runs",
                     method="update",
                     args=[{"visible": step_visibility}])
            )

        # Show/hide all runs
        buttons.append(
            dict(label="Show All Runs",
                 method="update",
                 args=[{"visible": [True] * len(fig.data)}])
        )
        buttons.append(
            dict(label="Hide All Runs",
                 method="update",
                 args=[{"visible": [False] * len(fig.data)}])
        )

        # Add the dropdown to the layout
        fig.update_layout(
            updatemenus=[
                dict(
                    type="dropdown",
                    direction="down",
                    showactive=True,
                    buttons=buttons,
                    x=1.15,
                    y=1.15
                )
            ]
        )

        return fig
    
    def create_oracle_insights_summary(self, insights: dict) -> Figure:
        """Create a summary table showing Oracle decision insights."""
        # Create summary data for the table
        summary_data = [
            {'Metric': 'Total Oracle Decisions', 'Value': str(insights['total_decisions'])},
            {'Metric': 'Information Sent', 'Value': f"{insights['information_sent_count']} ({insights['information_sent_percentage']:.1f}%)"},
            {'Metric': 'No Information Sent', 'Value': str(insights['no_information_count'])},
            {'Metric': 'Average Benefit When Sending', 'Value': f"{insights['average_benefit_when_sending']:.3f}"},
            {'Metric': 'Average Confidence', 'Value': f"{insights['average_confidence']:.3f}"},
            {'Metric': 'Most Common Decision Reason', 'Value': insights['most_common_reason']}
        ]
        
        # Create table figure
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Metric', 'Value'],
                fill_color='lightblue',
                align='left',
                font=dict(size=14, color='black')
            ),
            cells=dict(
                values=[[row['Metric'] for row in summary_data], 
                       [row['Value'] for row in summary_data]],
                fill_color='white',
                align='left',
                font=dict(size=12)
            )
        )])
        
        fig.update_layout(
            title="Oracle Decision-Making Insights Summary",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
