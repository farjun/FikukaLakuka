from dash import dash_table, Dash, dcc, html, Input, Output, State
from base64 import b64encode
import io
import warnings
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
import os
from pathlib import Path

# Add the project root to the path for imports
project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
sys.path.insert(0, project_root)

from Multi_Agent_Robot.multi_agent_robot.data.plots.figures_gen import PlotsGenerator
from Multi_Agent_Robot.multi_agent_robot.data.process.process_data import PlotlyDatabaseProcessor
from Multi_Agent_Robot.multi_agent_robot.data.plots.board_gen import (
    create_board_visualization, 
    create_agent_course_visualization,
    get_step_info,
    get_beliefs_info
)
import config

DBS_FOLDER_SAVED = Path(__file__).parent.parent / "runs"
app = Dash(__name__)

# Initialize processor
processor = PlotlyDatabaseProcessor()

# Define columns for main data table and simulation data table
columns_to_display = ['step', 'action', 'oracle_action', 'reward', 'observation', 'agent_rock_beliefs', 'oracle_beliefs']
simulation_columns = ['step', 'action', 'reward', 'observation', 'agent_rock_beliefs', 'oracle_beliefs']

# Enhanced Oracle analysis columns
oracle_analysis_columns = ['step', 'oracle_action', 'decision_reason', 'expected_benefit', 'confidence', 'agent_action', 'reward']

# Initialize a buffer for HTML export
buffer = io.StringIO()

# Generate and organize figures, main tables, and simulation tables by game
def get_encoded_html_and_figures():
    all_data = processor.load_data()

    # Dictionaries for main tables, figures, and simulation data by game
    figures_by_game = {}
    tables_by_game = {}
    simulations_by_game = {}

    for game_data in all_data:
        game_name = game_data['game_name']
        # Get figures, main table, and simulation tables for the game
        game_figures = PlotsGenerator().get_figures(game_data)
        game_table = processor.get_columns_table(game_data, columns_to_display)
        game_simulations = processor.get_simulation_data(game_data, simulation_columns)

        figures_by_game[game_name] = game_figures
        tables_by_game[game_name] = game_table
        simulations_by_game[game_name] = game_simulations

    # Write each figure as HTML to buffer for download
    for game, figures in figures_by_game.items():
        for fig in figures:
            fig.write_html(buffer)

    html_bytes = buffer.getvalue().encode()
    encoded = b64encode(html_bytes).decode()
    return encoded, figures_by_game, tables_by_game, simulations_by_game

# Get encoded HTML, figures, main tables, and simulations organized by game
encoded, figures_by_game, tables_by_game, simulations_by_game = get_encoded_html_and_figures()

# Create the app layout with tabs for each game, a collapsible simulation data table, and a refresh button
app.layout = html.Div([
    html.H4('Game Plots, Data, and Simulations by Tabs with Export Option'),
    html.P("↓↓↓ Download all plots as HTML ↓↓↓", style={"text-align": "right", "font-weight": "bold"}),

    # Tabs for each game, each containing its respective figures, main data table, and simulation tables
    dcc.Tabs(id="game-tabs", children=[
        dcc.Tab(label=game_name, children=[
            # Board Visualization Section
            html.H3(f"🎯 Game Board Visualization - {game_name}", 
                    style={"color": "darkgreen", "textAlign": "center", "marginBottom": "20px"}),
            
            # Board visualization
            html.Div([
                html.H4("Current Board State", style={"color": "darkgreen"}),
                html.Div([
                    html.Label("Select Step: "),
                    dcc.Dropdown(
                        id=f"{game_name}-step-selector",
                        options=[{"label": f"Step {i}", "value": i} for i in range(len(tables_by_game[game_name]))],
                        value=len(tables_by_game[game_name]) - 1 if len(tables_by_game[game_name]) > 0 else 0,
                        style={"width": "200px", "display": "inline-block", "marginLeft": "10px"}
                    )
                ], style={"marginBottom": "15px"}),
                html.Div(id=f"{game_name}-step-info", style={"marginBottom": "15px", "fontWeight": "bold"}),
                dcc.Graph(id=f"{game_name}-board-graph", 
                         figure=create_board_visualization({"game_name": game_name, "env_data": tables_by_game[game_name]}))
            ], style={"margin-bottom": "30px", "border": "2px solid darkgreen", "padding": "15px", "borderRadius": "10px"}),
            
            # Game Statistics Summary
            html.Div([
                html.H4("Game Statistics Summary", style={"color": "darkgreen"}),
                html.Div(id=f"{game_name}-stats-summary", children=[
                    html.P(f"Total Steps: {len(tables_by_game[game_name])}"),
                    html.P(f"Total Reward: {sum([row.get('reward', 0) for row in tables_by_game[game_name]]) if tables_by_game[game_name] else 0:.2f}"),
                    html.P(f"Rocks Collected: {sum([1 for row in tables_by_game[game_name] if row.get('action', '').startswith('COLLECT_ROCK')]) if tables_by_game[game_name] else 0}"),
                    html.P(f"Information Bought: {sum([1 for row in tables_by_game[game_name] if row.get('action', '').startswith('BUY_INFORMATION')]) if tables_by_game[game_name] else 0}")
                ])
            ], style={"margin-bottom": "30px", "border": "2px solid darkgreen", "padding": "15px", "borderRadius": "10px"}),
            
            # Agent Beliefs Summary
            html.Div([
                html.H4("Current Agent Beliefs", style={"color": "darkgreen"}),
                html.Div(id=f"{game_name}-beliefs-summary", children=[
                    html.P("Select a step to view agent beliefs about rocks")
                ])
            ], style={"margin-bottom": "30px", "border": "2px solid darkgreen", "padding": "15px", "borderRadius": "10px"}),
            
            # Agent Course Visualization
            html.Div([
                html.H4("Agent Course Throughout Game", style={"color": "darkgreen"}),
                dcc.Graph(id=f"{game_name}-course-graph", 
                         figure=create_agent_course_visualization({"game_name": game_name, "env_data": tables_by_game[game_name]}))
            ], style={"margin-bottom": "30px", "border": "2px solid darkgreen", "padding": "15px", "borderRadius": "10px"}),
            
            # Oracle Analysis Section
            html.H3(f"🔮 Oracle Decision Analysis - {game_name}", 
                    style={"color": "purple", "textAlign": "center", "marginBottom": "20px"}),
            
            # Oracle-specific plots (first 5 are Oracle analysis plots)
            html.Div([
                html.H4("Oracle Decision Analysis Plots", style={"color": "purple"})
            ] + [
                dcc.Graph(id=f"{game_name}-oracle-graph-{i}", figure=fig) 
                for i, fig in enumerate(figures[:5])  # First 5 are Oracle analysis
            ], style={"margin-bottom": "30px", "border": "2px solid purple", "padding": "15px", "borderRadius": "10px"}),
            
            # Traditional plots
            html.Div([
                html.H4("Traditional Analysis Plots", style={"color": "blue"})
            ] + [
                dcc.Graph(id=f"{game_name}-graph-{i}", figure=fig) 
                for i, fig in enumerate(figures[5:])  # Rest are traditional plots
            ], style={"margin-bottom": "30px", "border": "2px solid blue", "padding": "15px", "borderRadius": "10px"}),
            
            # Main data table
            html.H4("Main Game Data", style={"color": "green"}),
            dash_table.DataTable(
                id=f"{game_name}-table",
                columns=[{"name": col, "id": col} for col in columns_to_display],
                data=tables_by_game[game_name],
                style_header={
                    'textAlign': 'left'
                },
                style_data={
                    'whiteSpace': 'normal',
                    'height': 'auto',
                },
                style_cell={
                    'textAlign': 'left',
                    'padding': '10px',
                    'minWidth': '100px'
                }
            ),

            # Collapsible list of simulation tables
            html.H5(f"🔬 {game_name} Oracle Simulations", style={"color": "orange"}),
            html.Div([
                html.Div([
                    html.Button(f"Show/Hide Simulation {sim_index + 1}",
                                id=f"{game_name}-toggle-simulation-{sim_index}",
                                style={"backgroundColor": "orange", "color": "white", "border": "none", "padding": "10px", "margin": "5px", "borderRadius": "5px"}),
                    html.Div(
                        dash_table.DataTable(
                            id=f"{game_name}-simulation-table-{sim_index}",
                            columns=[{"name": col, "id": col} for col in simulation_columns],
                            data=sim_table,
                            style_cell={
                                'textAlign': 'left',
                                'padding': '8px',
                                'minWidth': '80px'
                            }
                        ),
                        id=f"{game_name}-simulation-div-{sim_index}",
                        style={"display": "none"}  # Initially collapsed
                    )
                ], style={"margin-bottom": "10px"}) for sim_index, sim_table in
                enumerate(simulations_by_game[game_name])
            ])
        ],
        style={
            "padding": "10px",  # Add padding inside the tab
            "whiteSpace": "normal",  # Allow wrapping for long names
            "minWidth": "150px",  # Set a minimum width for all tabs
            "flexGrow": "1",  # Let all tabs expand evenly
            "textAlign": "center",  # Center align the text
        }
        ) for game_name, figures in figures_by_game.items()
    ]),

    # Download link
    html.A(
        html.Button("Download all as HTML"),
        id="download",
        href="data:text/html;base64," + encoded,
        download="plotly_graphs.html"
    ),

    # Refresh button
    html.Button("Refresh Data", id="refresh-button")
])

# Callback to refresh data and update the store
@app.callback(
    Output("game-tabs", "children"),
    Input("refresh-button", "n_clicks")
)
def refresh_data(n_clicks):
    if n_clicks:
        # Reload all data when refresh button is clicked
        _, new_figures_by_game, new_tables_by_game, new_simulations_by_game = get_encoded_html_and_figures()
        
        # Update the global variables
        global figures_by_game, tables_by_game, simulations_by_game
        figures_by_game = new_figures_by_game
        tables_by_game = new_tables_by_game
        simulations_by_game = new_simulations_by_game
        
        # Return the new layout
        return [
            dcc.Tab(label=game_name, children=[
                # Board Visualization Section
                html.H3(f"🎯 Game Board Visualization - {game_name}", 
                        style={"color": "darkgreen", "textAlign": "center", "marginBottom": "20px"}),
                
                # Board visualization
                html.Div([
                    html.H4("Current Board State", style={"color": "darkgreen"}),
                    html.Div([
                        html.Label("Select Step: "),
                        dcc.Dropdown(
                            id=f"{game_name}-step-selector",
                            options=[{"label": f"Step {i}", "value": i} for i in range(len(new_tables_by_game[game_name]))],
                            value=len(new_tables_by_game[game_name]) - 1 if len(new_tables_by_game[game_name]) > 0 else 0,
                            style={"width": "200px", "display": "inline-block", "marginLeft": "10px"}
                        )
                    ], style={"marginBottom": "15px"}),
                    html.Div(id=f"{game_name}-step-info", style={"marginBottom": "15px", "fontWeight": "bold"}),
                    dcc.Graph(id=f"{game_name}-board-graph", 
                             figure=create_board_visualization({"game_name": game_name, "env_data": new_tables_by_game[game_name]}))
                ], style={"margin-bottom": "30px", "border": "2px solid darkgreen", "padding": "15px", "borderRadius": "10px"}),
                
                # Game Statistics Summary
                html.Div([
                    html.H4("Game Statistics Summary", style={"color": "darkgreen"}),
                    html.Div([
                        html.P(f"Total Steps: {len(new_tables_by_game[game_name])}"),
                        html.P(f"Total Reward: {sum([row.get('reward', 0) for row in new_tables_by_game[game_name]]) if new_tables_by_game[game_name] else 0:.2f}"),
                        html.P(f"Rocks Collected: {sum([1 for row in new_tables_by_game[game_name] if row.get('action', '').startswith('COLLECT_ROCK')]) if new_tables_by_game[game_name] else 0}"),
                        html.P(f"Information Bought: {sum([1 for row in new_tables_by_game[game_name] if row.get('action', '').startswith('BUY_INFORMATION')]) if new_tables_by_game[game_name] else 0}")
                    ])
                ], style={"margin-bottom": "30px", "border": "2px solid darkgreen", "padding": "15px", "borderRadius": "10px"}),
                
                # Agent Beliefs Summary
                html.Div([
                    html.H4("Current Agent Beliefs", style={"color": "darkgreen"}),
                    html.Div(id=f"{game_name}-beliefs-summary", children=[
                        html.P("Select a step to view agent beliefs about rocks")
                    ])
                ], style={"margin-bottom": "30px", "border": "2px solid darkgreen", "padding": "15px", "borderRadius": "10px"}),
                
                # Agent Course Visualization
                html.Div([
                    html.H4("Agent Course Throughout Game", style={"color": "darkgreen"}),
                    dcc.Graph(id=f"{game_name}-course-graph", 
                             figure=create_agent_course_visualization({"game_name": game_name, "env_data": new_tables_by_game[game_name]}))
                ], style={"margin-bottom": "30px", "border": "2px solid darkgreen", "padding": "15px", "borderRadius": "10px"}),
                
                # Oracle Analysis Section
                html.H3(f"🔮 Oracle Decision Analysis - {game_name}", 
                        style={"color": "purple", "textAlign": "center", "marginBottom": "20px"}),
                
                # Oracle-specific plots (first 5 are Oracle analysis plots)
                html.Div([
                    html.H4("Oracle Decision Analysis Plots", style={"color": "purple"})
                ] + [
                    dcc.Graph(id=f"{game_name}-oracle-graph-{i}", figure=fig) 
                    for i, fig in enumerate(figures[:5])  # First 5 are Oracle analysis
                ], style={"margin-bottom": "30px", "border": "2px solid purple", "padding": "15px", "borderRadius": "10px"}),
                
                # Traditional plots
                html.Div([
                    html.H4("Traditional Analysis Plots", style={"color": "blue"})
                ] + [
                    dcc.Graph(id=f"{game_name}-graph-{i}", figure=fig) 
                    for i, fig in enumerate(figures[5:])  # Rest are traditional plots
                ], style={"margin-bottom": "30px", "border": "2px solid blue", "padding": "15px", "borderRadius": "10px"}),
                
                # Main data table
                html.H4("Main Game Data", style={"color": "green"}),
                dash_table.DataTable(
                    id=f"{game_name}-table",
                    columns=[{"name": col, "id": col} for col in columns_to_display],
                    data=new_tables_by_game[game_name],
                    style_header={
                        'textAlign': 'left'
                    },
                    style_data={
                        'whiteSpace': 'normal',
                        'height': 'auto',
                    },
                    style_cell={
                        'textAlign': 'left',
                        'padding': '10px',
                        'minWidth': '100px'
                    }
                ),

                # Collapsible list of simulation tables
                html.H5(f"🔬 {game_name} Oracle Simulations", style={"color": "orange"}),
                html.Div([
                    html.Div([
                        html.Button(f"Show/Hide Simulation {sim_index + 1}",
                                    id=f"{game_name}-toggle-simulation-{sim_index}",
                                    style={"backgroundColor": "orange", "color": "white", "border": "none", "padding": "10px", "margin": "5px", "borderRadius": "5px"}),
                        html.Div(
                            dash_table.DataTable(
                                id=f"{game_name}-simulation-table-{sim_index}",
                                columns=[{"name": col, "id": col} for col in simulation_columns],
                                data=sim_table,
                                style_cell={
                                    'textAlign': 'left',
                                    'padding': '8px',
                                    'minWidth': '80px'
                                }
                            ),
                            id=f"{game_name}-simulation-div-{sim_index}",
                            style={"display": "none"}  # Initially collapsed
                        )
                    ], style={"margin-bottom": "10px"}) for sim_index, sim_table in
                    enumerate(new_simulations_by_game[game_name])
                ])
            ]) for game_name, figures in new_figures_by_game.items()
        ]
    
    # Return the current layout if no refresh
    return [
        dcc.Tab(label=game_name, children=[
            # Board Visualization Section
            html.H3(f"🎯 Game Board Visualization - {game_name}", 
                    style={"color": "darkgreen", "textAlign": "center", "marginBottom": "20px"}),
            
            # Board visualization
            html.Div([
                html.H4("Current Board State", style={"color": "darkgreen"}),
                html.Div([
                    html.Label("Select Step: "),
                    dcc.Dropdown(
                        id=f"{game_name}-step-selector",
                        options=[{"label": f"Step {i}", "value": i} for i in range(len(tables_by_game[game_name]))],
                        value=len(tables_by_game[game_name]) - 1 if len(tables_by_game[game_name]) > 0 else 0,
                        style={"width": "200px", "display": "inline-block", "marginLeft": "10px"}
                    )
                ], style={"marginBottom": "15px"}),
                html.Div(id=f"{game_name}-step-info", style={"marginBottom": "15px", "fontWeight": "bold"}),
                dcc.Graph(id=f"{game_name}-board-graph", 
                         figure=create_board_visualization({"game_name": game_name, "env_data": tables_by_game[game_name]}))
            ], style={"margin-bottom": "30px", "border": "2px solid darkgreen", "padding": "15px", "borderRadius": "10px"}),
            
            # Game Statistics Summary
            html.Div([
                html.H4("Game Statistics Summary", style={"color": "darkgreen"}),
                html.Div([
                    html.P(f"Total Steps: {len(tables_by_game[game_name])}"),
                    html.P(f"Total Reward: {sum([row.get('reward', 0) for row in tables_by_game[game_name]]) if tables_by_game[game_name] else 0:.2f}"),
                    html.P(f"Rocks Collected: {sum([1 for row in tables_by_game[game_name] if row.get('action', '').startswith('COLLECT_ROCK')]) if tables_by_game[game_name] else 0}"),
                    html.P(f"Information Bought: {sum([1 for row in tables_by_game[game_name] if row.get('action', '').startswith('BUY_INFORMATION')]) if tables_by_game[game_name] else 0}")
                ])
            ], style={"margin-bottom": "30px", "border": "2px solid darkgreen", "padding": "15px", "borderRadius": "10px"}),
            
            # Agent Beliefs Summary
            html.Div([
                html.H4("Current Agent Beliefs", style={"color": "darkgreen"}),
                html.Div(id=f"{game_name}-beliefs-summary", children=[
                    html.P("Select a step to view agent beliefs about rocks")
                ])
            ], style={"margin-bottom": "30px", "border": "2px solid darkgreen", "padding": "15px", "borderRadius": "10px"}),
            
            # Agent Course Visualization
            html.Div([
                html.H4("Agent Course Throughout Game", style={"color": "darkgreen"}),
                dcc.Graph(id=f"{game_name}-course-graph", 
                         figure=create_agent_course_visualization({"game_name": game_name, "env_data": tables_by_game[game_name]}))
            ], style={"margin-bottom": "30px", "border": "2px solid darkgreen", "padding": "15px", "borderRadius": "10px"}),
            
            # Oracle Analysis Section
            html.H3(f"🔮 Oracle Decision Analysis - {game_name}", 
                    style={"color": "purple", "textAlign": "center", "marginBottom": "20px"}),
            
            # Oracle-specific plots (first 5 are Oracle analysis plots)
            html.Div([
                html.H4("Oracle Decision Analysis Plots", style={"color": "purple"})
            ] + [
                dcc.Graph(id=f"{game_name}-oracle-graph-{i}", figure=fig) 
                for i, fig in enumerate(figures[:5])  # First 5 are Oracle analysis
            ], style={"margin-bottom": "30px", "border": "2px solid purple", "padding": "15px", "borderRadius": "10px"}),
            
            # Traditional plots
            html.Div([
                html.H4("Traditional Analysis Plots", style={"color": "blue"})
            ] + [
                dcc.Graph(id=f"{game_name}-graph-{i}", figure=fig) 
                for i, fig in enumerate(figures[5:])  # Rest are traditional plots
            ], style={"margin-bottom": "30px", "border": "2px solid blue", "padding": "15px", "borderRadius": "10px"}),
            
            # Main data table
            html.H4("Main Game Data", style={"color": "green"}),
            dash_table.DataTable(
                id=f"{game_name}-table",
                columns=[{"name": col, "id": col} for col in columns_to_display],
                data=tables_by_game[game_name],
                style_header={
                    'textAlign': 'left'
                },
                style_data={
                    'whiteSpace': 'normal',
                    'height': 'auto',
                },
                style_cell={
                    'textAlign': 'left',
                    'padding': '10px',
                    'minWidth': '100px'
                }
            ),

            # Collapsible list of simulation tables
            html.H5(f"🔬 {game_name} Oracle Simulations", style={"color": "orange"}),
            html.Div([
                html.Div([
                    html.Button(f"Show/Hide Simulation {sim_index + 1}",
                                id=f"{game_name}-toggle-simulation-{sim_index}",
                                style={"backgroundColor": "orange", "color": "white", "border": "none", "padding": "10px", "margin": "5px", "borderRadius": "5px"}),
                    html.Div(
                        dash_table.DataTable(
                            id=f"{game_name}-simulation-table-{sim_index}",
                            columns=[{"name": col, "id": col} for col in simulation_columns],
                            data=sim_table,
                            style_cell={
                                'textAlign': 'left',
                                'padding': '8px',
                                'minWidth': '80px'
                            }
                        ),
                        id=f"{game_name}-simulation-div-{sim_index}",
                        style={"display": "none"}  # Initially collapsed
                    )
                ], style={"margin-bottom": "10px"}) for sim_index, sim_table in
                enumerate(simulations_by_game[game_name])
            ])
        ]) for game_name, figures in figures_by_game.items()
    ]

# Callback to update board visualization when step is selected
for game_name, game_data in tables_by_game.items():
    app.callback(
        Output(f"{game_name}-board-graph", "figure"),
        Input(f"{game_name}-step-selector", "value")
    )(lambda step_idx, g_name=game_name: create_board_visualization(
        {"game_name": g_name, "env_data": tables_by_game[g_name]}, 
        tables_by_game[g_name][step_idx] if step_idx is not None and step_idx < len(tables_by_game[g_name]) else None
    ))
    
    # Callback to update step info display
    app.callback(
        Output(f"{game_name}-step-info", "children"),
        Input(f"{game_name}-step-selector", "value")
    )(lambda step_idx, g_name=game_name: get_step_info(tables_by_game[g_name], step_idx))
    
    # Callback to update beliefs summary
    app.callback(
        Output(f"{game_name}-beliefs-summary", "children"),
        Input(f"{game_name}-step-selector", "value")
    )(lambda step_idx, g_name=game_name: get_beliefs_info(tables_by_game[g_name], step_idx))

# Callbacks for toggling visibility of each simulation table
for game_name, simulations in simulations_by_game.items():
    for sim_index in range(len(simulations)):
        app.callback(
            Output(f"{game_name}-simulation-div-{sim_index}", "style"),
            Input(f"{game_name}-toggle-simulation-{sim_index}", "n_clicks"),
            State(f"{game_name}-simulation-div-{sim_index}", "style")
        )(lambda n, style: {"display": "block"} if n and style["display"] == "none" else {"display": "none"} if n else style)

if __name__ == "__main__":
    app.run(debug=True)
