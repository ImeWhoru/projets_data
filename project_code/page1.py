###################################################
##### IMPORTS AND FILES LOCATION              #####
###################################################

import pandas as pd
import math
import os
from dash import Dash, html, dcc, Input, Output

# Define paths
pathtocsv = 'celeba/'
image_folder = 'project_code/assets/img_celeba/'

# Safely load CSV files
csv_s = pathtocsv + 'celeba_buffalo_s.csv'
csv_l = pathtocsv + 'celeba_buffalo_l.csv'

try:
    df_s = pd.read_csv(csv_s, engine='python', encoding='utf-8')  # InsightFace small
    df_l = pd.read_csv(csv_l, engine='python', encoding='utf-8')  # InsightFace large
except Exception as e:
    print(f"Error reading CSV files: \n{e}")
    exit()

# Clean and process data
df_s_pg1 = df_s.iloc[:, :40].dropna()
df_l_pg1 = df_l.iloc[:, :40].dropna()

###################################################
##### BUILD THE DASH APP                        ####
###################################################

# Initialize Dash app
app = Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("Data Visualization with Dash", style={'textAlign': 'center'}),

    # Toggle-style dataset selector
    html.Div([
        dcc.RadioItems(
            id='dataset-selector',
            options=[
                {'label': 'Small', 'value': 'small'},
                {'label': 'Large', 'value': 'large'}
            ],
            value='small',  # Default value
            style={'display': 'none'}  # Hide default RadioItems
        ),
        html.Div(
            id="custom-toggle",
            style={
                'display': 'inline-block',
                'border': '1px solid black',
                'border-radius': '25px',
                'overflow': 'hidden',
                'width': '200px',
                'cursor': 'pointer',
                'margin': '20px auto',
                'textAlign': 'center'
            },
            children=[
                html.Div("Small", id="small-btn", style={
                    'width': '50%',
                    'display': 'inline-block',
                    'padding': '10px',
                    'color': 'white',
                    'background-color': 'green',
                    'cursor': 'pointer'
                }),
                html.Div("Large", id="large-btn", style={
                    'width': '50%',
                    'display': 'inline-block',
                    'padding': '10px',
                    'color': 'white',
                    'background-color': 'red',
                    'cursor': 'pointer'
                }),
            ]
        )
    ], style={'textAlign': 'center'}),

    # Characteristics Dropdown and Image Grid
    html.Div([
        # Characteristics Dropdown
        html.Div([
            html.Label("Filter Characteristics (Include/Exclude):"),
            dcc.Dropdown(
                id='characteristics-dropdown',
                multi=True,
                placeholder="Select Characteristics"
            ),
            dcc.RadioItems(
                id='filter-mode',
                options=[
                    {'label': 'Include', 'value': 'include'},
                    {'label': 'Exclude', 'value': 'exclude'}
                ],
                value='include',
                style={'margin-top': '10px'}
            )
        ], style={'width': '25%', 'float': 'right', 'margin': '20px'}),

        # Grid for images
        html.Div(id='visualization-box', style={
            'width': '70%',
            'float': 'left',
            'display': 'grid',
            'gridTemplateColumns': 'repeat(4, 1fr)',
            'gridGap': '10px',
            'padding': '10px',
            'border': '5px solid black'
        }),
    ], style={'clear': 'both'}),

    # Navigation Buttons
    html.Div([
        html.Button("Previous", id='prev-button', n_clicks=0, style={'margin-right': '10px'}),
        html.Button("Next", id='next-button', n_clicks=0)
    ], style={'textAlign': 'right', 'margin-top': '10px', 'margin-right': '15%'}),
])

###################################################
##### Page 2                                  #####
###################################################

from sklearn.manifold import TSNE

@app.callback(
    Output('tsne-plot', 'figure'),
    [Input('dataset-selector', 'value')]
)
def tsne_dimension_reduction(dataset):
    df = df_s_pg1 if dataset == 'small' else df_l_pg1
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(df.iloc[:, 1:])  # Assuming first column is not a feature
    
    fig = px.scatter(
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        labels={'x': 'TSNE-1', 'y': 'TSNE-2'},
        title='t-SNE Dimension Reduction'
    )
    return fig

###################################################
##### CALLBACKS FOR INTERACTIVITY             #####
###################################################

# Toggle Callback
@app.callback(
    [Output('small-btn', 'style'),
     Output('large-btn', 'style')],
    Input('dataset-selector', 'value')
)
def update_toggle(dataset):
    selected_style = {'width': '50%', 'padding': '10px', 'color': 'white', 'background-color': 'green', 'cursor': 'pointer'}
    unselected_style = {'width': '50%', 'padding': '10px', 'color': 'white', 'background-color': 'red', 'cursor': 'pointer'}

    if dataset == 'small':
        return selected_style, unselected_style
    else:
        return unselected_style, selected_style


@app.callback(
    Output('dataset-selector', 'value'),
    [Input('small-btn', 'n_clicks'),
     Input('large-btn', 'n_clicks')],
    prevent_initial_call=True
)
def handle_toggle(small_clicks, large_clicks):
    ctx = Dash.callback_context
    if not ctx.triggered:
        return Dash.no_update
    clicked_id = ctx.triggered[0]['prop_id'].split('.')[0]
    return 'small' if clicked_id == 'small-btn' else 'large'


# Populate characteristics based on dataset
@app.callback(
    Output('characteristics-dropdown', 'options'),
    Input('dataset-selector', 'value')
)
def update_characteristics_options(selected_dataset):
    df = df_s_pg1 if selected_dataset == 'small' else df_l_pg1
    return [{'label': col, 'value': col} for col in df.columns[1:]]


# Visualization with Pagination
@app.callback(
    Output('visualization-box', 'children'),
    [Input('dataset-selector', 'value'),
     Input('characteristics-dropdown', 'value'),
     Input('filter-mode', 'value'),
     Input('prev-button', 'n_clicks'),
     Input('next-button', 'n_clicks')]
)
def update_visualization(dataset, characteristics, filter_mode, prev_clicks, next_clicks):
    df = df_s_pg1 if dataset == 'small' else df_l_pg1
    if characteristics:
        if filter_mode == 'include':
            filtered_df = df[(df[characteristics] == 1).all(axis=1)]
        else:  # Exclude
            filtered_df = df[(df[characteristics] == -1).all(axis=1)]
    else:
        filtered_df = df

    images = [f"/assets/img_celeba/{row.iloc[0]}" for _, row in filtered_df.iterrows()]
    page_size = 12
    total_pages = max(1, math.ceil(len(images) / page_size))
    current_page = (next_clicks - prev_clicks) % total_pages
    start_idx, end_idx = current_page * page_size, (current_page + 1) * page_size
    return [html.Img(src=img, style={'width': '100%', 'padding': '5px'}) for img in images[start_idx:end_idx]]


###################################################
##### RUN THE APP                              #####
###################################################

if __name__ == '__main__':
    app.run_server(debug=True)
