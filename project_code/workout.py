# workout.py

###################################################
##### IMPORTS AND FILES LOCATION + LOAD DATA  #####
###################################################

from dash import Dash, html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate

from functions import show_sidebar, get_theme_styles
from data_loader import load_datasets
from pages.page1 import render_page_1, register_page_1_callbacks
from pages.page2 import render_page_2

df_s_pg1, df_l_pg1 = load_datasets()

###################################################
##### INITIALIZE DASH APP                     #####
###################################################
# Initialize Dash app
app = Dash(__name__, suppress_callback_exceptions=True)

# Initial state for the app layout
initial_sidebar_style, initial_content_style, _, initial_sidebar_content = show_sidebar(
    sidebar_state='expanded', theme='light', dataset='small'
)

app.layout = html.Div([
    dcc.Store(id='theme-store', data='light'),  # Default theme is light
    dcc.Store(id='dataset-store', data='small'),  # Default dataset
    dcc.Store(id='sidebar-state', data='expanded'),  # Sidebar is initially expanded

    # Sidebar container
    html.Div([
        html.Button("☰", id="toggle-button", style={
            'background-color': 'transparent',
            'border': 'none',
            'color': 'blue',
            'font-size': '24px',
            'cursor': 'pointer',
            'margin-bottom': '20px'
        }),
        html.Div(id='sidebar-content', children=initial_sidebar_content)
    ], id='sidebar', style=initial_sidebar_style),

    # Main content container
    html.Div(id='page-content', style=initial_content_style),

    dcc.Location(id='url', refresh=False)  # For page routing
])


###################################################
##### CALLBACKS                               #####
###################################################

# Update theme in theme-store
@app.callback(
    Output('theme-store', 'data'),
    Input('theme-toggle', 'value'),
    prevent_initial_call=True
)
def update_theme_store(theme: str) -> str:
    return theme

# Update dataset in dataset-store
@app.callback(
    Output('dataset-store', 'data'),
    Input('dataset-selector', 'value'),
    prevent_initial_call=True
)
def update_dataset_store(dataset: str) -> str:
    return dataset

@app.callback(
    [Output('page-content', 'children'),
     Output('page-content', 'style'),
     Output('sidebar', 'style'),
     Output('sidebar-content', 'children'),
     Output('sidebar-state', 'data')],
    [Input('url', 'pathname'),
     Input('theme-store', 'data'),
     Input('dataset-store', 'data'),
     Input('toggle-button', 'n_clicks')],
    [State('sidebar-state', 'data')]
)
def update_page_and_sidebar(pathname, theme, dataset, n_clicks, sidebar_state):
    """
    Combines the logic for rendering pages, updating sidebar styles, and handling theme changes.
    """
    # Determine sidebar state
    if n_clicks is None:
        n_clicks = 0
    if n_clicks % 2 == 1:
        sidebar_state = 'collapsed'
    else:
        sidebar_state = 'expanded'

    # Get the appropriate dataset
    data = df_s_pg1 if dataset == 'small' else df_l_pg1

    # Get updated sidebar and content styles
    sidebar_style, content_style, new_sidebar_state, sidebar_content = show_sidebar(sidebar_state, theme, dataset)

    # Render the requested page
    if pathname == '/page-2':
        page_content = render_page_2(theme, dataset, data)
    else:
        page_content = render_page_1(theme, dataset, data)

    return page_content, content_style, sidebar_style, sidebar_content, new_sidebar_state


###################################################
##### RUN THE APP                             #####
###################################################

if __name__ == '__main__':
    # Register callbacks before starting the server
    register_page_1_callbacks(app, df_s_pg1, df_l_pg1)

    # Start the Dash server
    app.run_server(debug=True)
