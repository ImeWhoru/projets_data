
###################################################
##### IMPORTS AND FILES LOCATION              #####
###################################################

import pandas as pd
import numpy as np
import os
import math
from dash import Dash, html, dcc, Input, Output  # Dash library

# Define paths
pathtocsv = 'projets_data/celeba/'
image_folder = pathtocsv + 'img_celeba/'

# Set for debugging and analysis
ANALYSE_DATA = False


###################################################
##### EXTRACT THE DATASET AND INITIAL ANALYSIS ####
###################################################

# Load the data from the CSV files
csv_s = pathtocsv + 'celeba_buffalo_s.csv'
csv_l = pathtocsv + 'celeba_buffalo_l.csv'

# Safely load CSVs
try:
    df_s = pd.read_csv(csv_s, engine='python', encoding='utf-8')  # InsightFace small
    df_l = pd.read_csv(csv_l, engine='python', encoding='utf-8')  # InsightFace large
except Exception as e:
    print(f"Error reading CSV files: {e}")
    exit()

# For the first page, only the first 40 columns are needed
df_s_pg1 = df_s.iloc[:, :40].dropna()
df_l_pg1 = df_l.iloc[:, :40].dropna()

if ANALYSE_DATA:
    # Check if columns are the same across datasets
    print("########################################")
    if df_s_pg1.columns.tolist() != df_l_pg1.columns.tolist():
        print("The columns are not the same.")
        exit()
    print("The columns are the same.")

    # Perform basic analysis
    print("########################################")
    print(f"Small dataset columns: {df_s_pg1.columns.tolist()}")
    print("First 10 rows of the small dataset:")
    print(df_s_pg1.head(10))
    print(f"The small dataset has {df_s_pg1.shape[0]} rows and {df_s_pg1.shape[1]} columns.")
    print("Column data types:")
    print(df_s_pg1.dtypes)
    print("Summary statistics:")
    print(df_s_pg1.describe(include='all'))


###################################################
##### BUILD THE DASH APP                        ####
###################################################

# Initialize the Dash app
app = Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("Data Visualization with Dash", style={'textAlign': 'center'}),
    
    # Dropdown to select dataset
    html.Div([
        html.Label("Select Dataset:"),
        dcc.Dropdown(
            id='dataset-selector',
            options=[
                {'label': 'InsightFace Small', 'value': 'small'},
                {'label': 'InsightFace Large', 'value': 'large'}
            ],
            value='small'
        )
    ], style={'width': '50%', 'margin': 'auto'}),
    
    # Main content: Grid + Checklist + Navigation
    html.Div([
        # Grid for images
        html.Div(id='visualization-box', style={
            'width': '50%', 'height': 'auto', 'border': '5px solid black',
            'float': 'left', 'display': 'grid', 'gridTemplateColumns': 'repeat(4, 1fr)',
            'gridGap': '10px', 'padding': '10px', 'box-sizing': 'border-box'
        }),

        # Characteristics dropdown
        html.Div([
            html.Label("Filter Characteristics (Include/Exclude):"),
            dcc.Dropdown(
                id='characteristics-dropdown',
                options=[
                    {'label': '5_o_Clock_Shadow', 'value': '5_o_Clock_Shadow'},
                    {'label': 'Arched_Eyebrows', 'value': 'Arched_Eyebrows'},
                    {'label': 'Bags_Under_Eyes', 'value': 'Bags_Under_Eyes'},
                    {'label': 'Bald', 'value': 'Bald'},
                    # Add all other characteristics here
                ],
                multi=True,
                placeholder="Select Characteristics",
            ),
            dcc.RadioItems(
                id='filter-mode',
                options=[
                    {'label': 'Include', 'value': 'include'},
                    {'label': 'Exclude', 'value': 'exclude'}
                ],
                value='include',  # Default to include mode
                style={'margin-top': '10px'}
            )
        ], style={'width': '25%', 'float': 'right', 'margin': '20px'}),
    ], style={'overflow': 'hidden', 'clear': 'both'}),

    # Navigation buttons below the grid
    html.Div([
        html.Button("Previous", id='prev-button', n_clicks=0, style={'margin-right': '10px'}),
        html.Button("Next", id='next-button', n_clicks=0)
    ], style={'textAlign': 'right', 'margin-top': '10px', 'margin-right': '15%'})
])


###################################################
##### CALLBACKS FOR INTERACTIVITY             #####
###################################################

# Populate checklist options based on dataset
@app.callback(
    Output('characteristics-dropdown', 'options'),
    Input('dataset-selector', 'value')
)
def update_characteristics_options(selected_dataset):
    if selected_dataset == 'small':
        return [{'label': col, 'value': col} for col in df_s_pg1.columns[1:]]
    else:
        return [{'label': col, 'value': col} for col in df_l_pg1.columns[1:]]

# Visualization logic with pagination
@app.callback(
    Output('visualization-box', 'children'),
    [Input('dataset-selector', 'value'),
     Input('characteristics-dropdown', 'value'),
     Input('filter-mode', 'value'),
     Input('prev-button', 'n_clicks'),
     Input('next-button', 'n_clicks')]
)
def update_visualization(selected_dataset, selected_characteristics, filter_mode, prev_clicks, next_clicks):
    if selected_characteristics and len(selected_characteristics) > 10:
        return html.Div("Please select no more than 10 characteristics.", style={'color': 'red'})

    # Select dataset
    df = df_s_pg1 if selected_dataset == 'small' else df_l_pg1

    # Apply filtering based on "Include" or "Exclude"
    if selected_characteristics:
        if filter_mode == 'include':
            filtered_df = df[(df[selected_characteristics] == 1).all(axis=1)]
        elif filter_mode == 'exclude':
            filtered_df = df[(df[selected_characteristics] == -1).all(axis=1)]
    else:
        filtered_df = df  # No filtering if no characteristics are selected

    # Generate list of image paths
    images = []
    for _, row in filtered_df.iterrows():
        image_filename = str(row.iloc[0]).strip()
        image_path = f"/assets/img_celeba/{image_filename}"
        images.append(image_path)

    # Handle no matches
    if not images:
        return html.Div("No matching images found.", style={'color': 'red'})

    # Pagination logic
    page_size = 12  # Number of images per page
    total_pages = max(1, math.ceil(len(images) / page_size))
    current_page = (next_clicks - prev_clicks) % total_pages
    start_idx = current_page * page_size
    end_idx = start_idx + page_size
    paginated_images = images[start_idx:end_idx]

    # Display images
    return [
        html.Img(src=img, style={
            'width': '100%', 'height': 'auto', 'object-fit': 'contain', 'padding': '5px'
        }) for img in paginated_images
    ]


###################################################
##### RUN THE APP                              ####
###################################################

if __name__ == '__main__':
    app.run_server(debug=True)
