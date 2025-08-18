from dash import dash_table, Dash, dcc, html, Input, Output, State
from base64 import b64encode
import io

from Multi_Agent_Robot.multi_agent_robot.data.plots.figures_gen import PlotsGenerator
from Multi_Agent_Robot.multi_agent_robot.data.process.process_data import PlotlyDatabaseProcessor
from pathlib import Path

# DBS_FOLDER_SAVED = Path(__file__).parent.parent / "runs"
DBS_FOLDER_SAVED = Path(__file__).parent.parent / "saved runs" / "15_01"
app = Dash(__name__)



# Initialize processor
processor = PlotlyDatabaseProcessor()

# Define columns for main data table and simulation data table
columns_to_display =['step', 'action', 'oracle_action', 'reward', 'observation', 'agent_rock_beliefs', 'oracle_beliefs']
simulation_columns = ['step', 'action', 'reward', 'observation', 'agent_rock_beliefs', 'oracle_beliefs']

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
            html.Div(
                [dcc.Graph(id=f"{game_name}-graph-{i}", figure=fig) for i, fig in enumerate(figures)],
                style={"margin-bottom": "20px"}
            ),
            dash_table.DataTable(
                id=f"{game_name}-table",
                columns=[{"name": col, "id": col} for col in columns_to_display],
                data=tables_by_game[game_name],
                style_header={
                    'textAlign': 'left'
                }
            ),

            # Collapsible list of simulation tables
            html.H5(f"{game_name} Simulations"),
            html.Div([
                html.Div([
                    html.Button(f"Show/Hide Simulation {sim_index + 1}",
                                id=f"{game_name}-toggle-simulation-{sim_index}"),
                    html.Div(
                        dash_table.DataTable(
                            id=f"{game_name}-simulation-table-{sim_index}",
                            columns=[{"name": col, "id": col} for col in simulation_columns],
                            data=sim_table
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
    html.Button("Refresh Data", id="refresh-button"),

    # Hidden div to store refreshed data
    dcc.Store(id="data-store",
              data={"figures": figures_by_game, "tables": tables_by_game, "simulations": simulations_by_game})
])


# Callback to refresh data and update the store
@app.callback(
    Output("data-store", "data"),
    Input("refresh-button", "n_clicks")
)
def refresh_data(n_clicks):
    if n_clicks:
        _, new_figures_by_game, new_tables_by_game, new_simulations_by_game = get_encoded_html_and_figures()
        return {"figures": new_figures_by_game, "tables": new_tables_by_game, "simulations": new_simulations_by_game}
    return {"figures": figures_by_game, "tables": tables_by_game, "simulations": simulations_by_game}


# Callback to update the layout with refreshed data
@app.callback(
    Output("game-tabs", "children"),
    Input("data-store", "data")
)
def update_tabs(data):
    new_figures_by_game = data["figures"]
    new_tables_by_game = data["tables"]
    new_simulations_by_game = data["simulations"]

    return [
        dcc.Tab(label=game_name, children=[
            html.Div(
                [dcc.Graph(id=f"{game_name}-graph-{i}", figure=fig) for i, fig in enumerate(figures)],
                style={"margin-bottom": "20px"}
            ),
            dash_table.DataTable(
                id=f"{game_name}-table",
                columns=[{"name": col, "id": col} for col in columns_to_display],
                data=new_tables_by_game[game_name]
            ),
            html.H5(f"{game_name} Simulations"),
            html.Div([
                html.Div([
                    html.Button(f"Show/Hide Simulation {sim_index + 1}",
                                id=f"{game_name}-toggle-simulation-{sim_index}"),
                    html.Div(
                        dash_table.DataTable(
                            id=f"{game_name}-simulation-table-{sim_index}",
                            columns=[{"name": col, "id": col} for col in simulation_columns],
                            data=sim_table
                        ),
                        id=f"{game_name}-simulation-div-{sim_index}",
                        style={"display": "none"}  # Initially collapsed
                    )
                ],
                style={"margin-bottom": "10px"}) for sim_index, sim_table in
                enumerate(new_simulations_by_game[game_name])
            ])
        ]) for game_name, figures in new_figures_by_game.items()
    ]


# Callbacks for toggling visibility of each simulation table
for game_name, simulations in simulations_by_game.items():
    for sim_index in range(len(simulations)):
        app.callback(
            Output(f"{game_name}-simulation-div-{sim_index}", "style"),
            Input(f"{game_name}-toggle-simulation-{sim_index}", "n_clicks"),
            State(f"{game_name}-simulation-div-{sim_index}", "style")
        )(lambda n, style: {"display": "block"} if n and style["display"] == "none" else {"display": "none"} if n else style)

if __name__ == "__main__":
    app.run_server(debug=True)


# metric to compute:
# 1. relation between buy information q value compared to reward difference with\without information
# 2. how much uncertenty divided by information gain there is in the enviroment vs
# 3. look at the complexity reduction after information gain

# 4 envs - make different entropy envs of 2,4,8,16
# 15-15 seed runs, with and without oracle.