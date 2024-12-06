###################################################
##### IMPORTS AND FILES LOCATION              #####
###################################################

import pandas as pd
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

###################################################
##### GLOBAL VARIABLES                        #####
###################################################

# Global variables to store theme and dataset state
global_theme = "light"  # Default theme
global_dataset = "small"  # Default dataset

###################################################
##### BUILD THE DASH APP                       #####
###################################################

# Initialize Dash app
app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),

    # Sidebar for toggles
    html.Div([
        html.H2("Settings", style={'textAlign': 'center', 'margin-bottom': '20px'}),

        # Theme Toggle
        html.Label("Theme Selector", style={'font-weight': 'bold'}),
        dcc.RadioItems(
            id='theme-toggle',
            options=[
                {'label': 'Light Mode', 'value': 'light'},
                {'label': 'Dark Mode', 'value': 'dark'}
            ],
            value='light',  # Default to Light Mode
            style={'margin-bottom': '20px'}
        ),

        # Dataset Toggle
        html.Label("Dataset Selector", style={'font-weight': 'bold'}),
        dcc.RadioItems(
            id='dataset-selector',
            options=[
                {'label': 'Small Dataset', 'value': 'small'},
                {'label': 'Large Dataset', 'value': 'large'}
            ],
            value='small',  # Default to Small Dataset
        ),

        # Navigation Links
        html.Hr(),
        html.A("Page 1", href="/page-1", style={'display': 'block', 'margin-top': '10px'}),
        html.A("Page 2", href="/page-2", style={'display': 'block', 'margin-top': '10px'})
    ], style={
        'width': '10%', 
        'position': 'fixed', 
        'height': '100%', 
        'padding': '20px', 
        'background-color': '#f4f4f4', 
        'box-shadow': '2px 0px 5px rgba(0,0,0,0.1)'
    }),

    # Main content area
    html.Div(id='page-content', style={
        'margin-left': '12%', 
        'padding': '20px',
        'background-color': '#f9f9f9',
        'min-height': '100vh'
    })
])

###################################################
##### CALLBACKS FOR INTERACTIVITY             #####
###################################################

# Update global theme and dataset from toggles
@app.callback(
    Output('theme-toggle', 'value'),
    Input('theme-toggle', 'value')
)
def update_global_theme(theme):
    global global_theme
    global_theme = theme
    return theme

@app.callback(
    Output('dataset-selector', 'value'),
    Input('dataset-selector', 'value')
)
def update_global_dataset(dataset):
    global global_dataset
    global_dataset = dataset
    return dataset

# Render the correct page based on the URL and dynamically update when theme or dataset changes
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname'),
     Input('theme-toggle', 'value'),
     Input('dataset-selector', 'value')]
)
def display_page(pathname, theme, dataset):
    global global_theme, global_dataset
    global_theme = theme
    global_dataset = dataset

    if pathname == '/page-2':
        return render_page_2(global_theme, global_dataset)
    else:  # Default to Page 1
        return render_page_1(global_theme, global_dataset)

# Update the page content style based on the theme
@app.callback(
    Output('page-content', 'style'),
    Input('theme-toggle', 'value')
)
def update_page_style(theme):
    if theme == 'dark':
        return {
            'margin-left': '12%',
            'padding': '20px',
            'background-color': '#333',
            'color': '#fff',
            'min-height': '100vh'
        }
    else:
        return {
            'margin-left': '12%',
            'padding': '20px',
            'background-color': '#f9f9f9',
            'color': '#000',
            'min-height': '100vh'
        }

###################################################
##### PAGE CONTENT DEFINITIONS                #####
###################################################

def render_page_1(theme, dataset):
    return html.Div([
        html.H1("Page 1", style={'textAlign': 'center'}),
        html.P(f"Theme: {theme.capitalize()}, Dataset: {dataset.capitalize()}."), 
        html.P("This is the content of Page 1.")
    ])

def render_page_2(theme, dataset):
    return html.Div([
        html.H1("Page 2", style={'textAlign': 'center'}),
        html.P(f"Theme: {theme.capitalize()}, Dataset: {dataset.capitalize()}."), 
        html.P("This is the content of Page 2.")
    ])

###################################################
##### RUN THE APP                             #####
###################################################

if __name__ == '__main__':
    app.run_server(debug=True)
