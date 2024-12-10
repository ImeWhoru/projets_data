import math
from dash import html, dcc, Input, Output, State, ctx
from dash.dependencies import ALL
from functions import get_theme_styles
import os
import cv2
import numpy as np
from tqdm import tqdm
import plotly.express as px
import pandas as pd

# Define the folder for images
image_folder = 'assets/img_celeba/'  # Dash serves from `assets/`
image_folder_average = 'celeba/img_celeba/'
output_path = 'project_code/assets/average_image.jpg'

# Render Page 1
def render_page_1(theme: str, dataset: str, data) -> html.Div:
    """
    Render Page 1 with characteristics filtering, image grid, and pagination.
    Args:
        theme (str): Current theme ('light' or 'dark').
        dataset (str): Current dataset selection ('small' or 'large').
        data: The dataset (DataFrame) to display.
    Returns:
        html.Div: Page layout.
    """
    theme_styles = get_theme_styles(theme)

    return html.Div([
        html.H1("Data Visualization with Dash", style={
            'textAlign': 'center',
            'color': theme_styles['color']
        }),

        # Characteristics Dropdown and Options
        html.Div([
            # Dropdown for selecting characteristics
            html.Div([
                html.Label("Filter Characteristics:", style={'color': theme_styles['color']}),
                dcc.Dropdown(
                    id='characteristics-dropdown',
                    multi=True,
                    placeholder="Select Characteristics",
                    style={'background-color': theme_styles['background-color'], 'color': theme_styles['color']}
                ),
            ], style={
                'margin-bottom': '10px'
            }),

            # Display selected characteristics with Include/Exclude options
            html.Div(id='selected-characteristics-container', style={
                'margin-top': '10px',
                'background-color': theme_styles['background-color'],
                'padding': '10px',
                'border-radius': '8px',
                'box-shadow': '0px 4px 6px rgba(0,0,0,0.2)'
            }),
        ], style={
            'width': '30%',
            'float': 'right',
            'margin': '20px',
            'background-color': theme_styles['background-color'],
            'padding': '10px',
            'border-radius': '8px',
            'box-shadow': '0px 4px 6px rgba(0,0,0,0.2)'
        }),

        # Display for counts
        html.Div([
            html.P(id='people-count', style={'font-weight': 'bold', 'margin-bottom': '10px'}),
            html.P(id='feature-count', style={'font-weight': 'bold', 'margin-bottom': '10px'}),
        ], style={
            'width': '60%',
            'float': 'left',
            'margin-bottom': '20px',
            'background-color': theme_styles['background-color'],
            'padding': '10px',
            'border-radius': '8px',
            'box-shadow': '0px 4px 6px rgba(0,0,0,0.2)'
        }),

        # Grid for images
        html.Div(id='visualization-box', style={
            'width': '60%',
            'float': 'left',
            'display': 'grid',
            'gridTemplateColumns': 'repeat(5, 1fr)',  # 5 columns
            'gridTemplateRows': 'repeat(2, auto)',  # 2 rows
            'gridGap': '5px',
            'padding': '5px',
            'border': f'3px solid {theme_styles["color"]}',
            'background-color': theme_styles['background-color']
        }),

        # Navigation Buttons
        html.Div([
            html.Button("Previous", id='prev-button', n_clicks=0, style={
                'margin-right': '10px',
                'background-color': theme_styles['background-color'],
                'color': theme_styles['color'],
                'border': f'2px solid {theme_styles["color"]}',
                'padding': '8px 16px',
                'border-radius': '5px'
            }),
            html.Button("Next", id='next-button', n_clicks=0, style={
                'background-color': theme_styles['background-color'],
                'color': theme_styles['color'],
                'border': f'2px solid {theme_styles["color"]}',
                'padding': '8px 16px',
                'border-radius': '5px'
            })
        ], style={'textAlign': 'right', 'margin-top': '10px', 'margin-right': '15%'}),

        # Button for generating the average image
        html.Button("Generate Average Picture", id='generate-average-btn', n_clicks=0, style={
            'margin': '20px 0',
            'padding': '10px',
            'background-color': theme_styles['background-color'],
            'color': theme_styles['color'],
            'border': f'2px solid {theme_styles["color"]}',
            'border-radius': '5px'
        }),

        # Placeholder for displaying the average image
        html.Div(id='average-image-container', children=[
            html.P("Average image will appear here.", style={'textAlign': 'center', 'color': theme_styles['color']}),
        ], style={
            'textAlign': 'center',
            'margin-top': '20px'
        }),

        # Attribute Distribution Bar Chart
        html.Div(id='attribute-distribution-graph', style={
            'textAlign': 'center',
            'margin-top': '30px',
            'padding': '20px',
            'width': '80%',
            'margin-left': 'auto',
            'margin-right': 'auto',
            'border-radius': '8px',
            'background-color': theme_styles['background-color'],
            'box-shadow': '0px 4px 6px rgba(0,0,0,0.2)'
        }),
    ], style={
        'background-color': theme_styles['background-color'],
        'color': theme_styles['color'],
        'padding': '20px'
    })


# Callbacks for Page 1
def register_page_1_callbacks(app, df_s_pg1, df_l_pg1):
    @app.callback(
        Output('characteristics-dropdown', 'options'),
        Input('dataset-store', 'data')
    )
    def update_characteristics_options(selected_dataset):
        df = df_s_pg1 if selected_dataset == 'small' else df_l_pg1
        characteristic_columns = df.columns[1:]  # Adjust this as per your dataset
        return [{'label': col, 'value': col} for col in characteristic_columns]

    @app.callback(
        [Output('selected-characteristics-container', 'children'),
         Output('feature-count', 'children')],
        Input('characteristics-dropdown', 'value')
    )
    def update_selected_characteristics(selected_characteristics):
        if not selected_characteristics:
            return html.P("No characteristics selected.", style={'color': 'gray', 'textAlign': 'center'}), "Number of Features Selected: 0"

        children = []
        for characteristic in selected_characteristics:
            children.append(html.Div([
                html.Label(f"{characteristic}:", style={'font-weight': 'bold', 'margin-right': '10px'}),
                dcc.RadioItems(
                    id={'type': 'filter-mode', 'index': characteristic},
                    options=[{'label': 'Include', 'value': 'include'}, {'label': 'Exclude', 'value': 'exclude'}],
                    value='include'
                )
            ]))
        feature_count = f"Number of Features Selected: {len(selected_characteristics)}"
        return children, feature_count

    @app.callback(
        [Output('visualization-box', 'children'),
         Output('people-count', 'children')],
        [Input('dataset-store', 'data'),
         Input({'type': 'filter-mode', 'index': ALL}, 'value'),
         Input('prev-button', 'n_clicks'),
         Input('next-button', 'n_clicks')],
        [State('characteristics-dropdown', 'value')]
    )
    def update_visualization(dataset, filter_modes, prev_clicks, next_clicks, selected_characteristics):
        df = df_s_pg1 if dataset == 'small' else df_l_pg1
        if selected_characteristics and filter_modes:
            for characteristic, mode in zip(selected_characteristics, filter_modes):
                df = df[df[characteristic] == 1] if mode == 'include' else df[df[characteristic] != 1]

        images = [f"{image_folder}{row.iloc[0]}" for _, row in df.iterrows()]
        people_count = f"People Found: {len(df)}"
        page_size = 10
        current_page = (next_clicks - prev_clicks) % (max(1, math.ceil(len(images) / page_size)))
        start_idx, end_idx = current_page * page_size, (current_page + 1) * page_size

        if not images:
            return [html.P("No images to display.", style={'textAlign': 'center', 'color': 'red'})], people_count

        return [html.Img(src=img, style={'width': '100%'}) for img in images[start_idx:end_idx]], people_count

    @app.callback(
        Output('average-image-container', 'children'),
        Input('generate-average-btn', 'n_clicks'),
        State('dataset-store', 'data'),
        State('characteristics-dropdown', 'value'),
        prevent_initial_call=True
    )
    def generate_average_image(n_clicks, dataset, selected_characteristics):
        df = df_s_pg1 if dataset == 'small' else df_l_pg1
        if selected_characteristics:
            for characteristic in selected_characteristics:
                df = df[df[characteristic] == 1]
        image_paths = [f"{image_folder_average}{img}" for img in df['image_name']]
        try:
            compute_average_image(image_paths, output_path, resize_dim=(256, 256))
            return html.Img(src=f'/assets/{os.path.basename(output_path)}', style={'max-width': '100%'})
        except ValueError as e:
            return html.P(str(e), style={'color': 'red'})

    @app.callback(
        Output('attribute-distribution-graph', 'children'),
        [Input('dataset-store', 'data')],
        [State('characteristics-dropdown', 'value')]
    )
    def update_distribution_graph(dataset, selected_characteristics):
        df = df_s_pg1 if dataset == 'small' else df_l_pg1
        if selected_characteristics:
            data = df[selected_characteristics].sum().reset_index()
            data.columns = ['Attribute', 'Count']
            fig = px.bar(data, x='Attribute', y='Count', title="Attribute Distribution")
            fig.update_layout(margin={'l': 0, 'r': 0, 't': 40, 'b': 0})
            return dcc.Graph(figure=fig)
        return html.P("Select characteristics to view the distribution.", style={'textAlign': 'center', 'color': 'gray'})


def compute_average_image(image_paths, output_path, resize_dim=None):
    total_images = 0
    sum_image = None
    for img_path in tqdm(image_paths, desc="Processing images"):
        img = cv2.imread(img_path)
        if img is None:
            continue
        if resize_dim:
            img = cv2.resize(img, resize_dim)
        img = img.astype(np.float32)
        if sum_image is None:
            sum_image = np.zeros_like(img)
        sum_image += img
        total_images += 1
    if total_images == 0:
        raise ValueError("No valid images found.")
    avg_image = np.clip(sum_image / total_images, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, avg_image)
