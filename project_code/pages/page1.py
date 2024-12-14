import math
from dash import html, dcc, Input, Output, State, ctx
from dash.dependencies import Input, Output, State, ALL

from dash.exceptions import PreventUpdate

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
    theme_styles = get_theme_styles(theme)

    return html.Div([

            # Box for the title
            html.Div([
                # Page 1 Header
                html.H1("Page 1: Data Visualization", 
                        style={
                            'textAlign': 'center',
                            'color': theme_styles['title-color'],
                            'font-size': '3em',
                        }),
            ], style={
                'background-color': theme_styles['titlebg-color'],
                'height': '100px',
                'width': '45%',
                'margin': 'auto',
                'display': 'flex',
                'align-items': 'center',
                'justify-content': 'center',
                'border-radius': '50px',
                'shadow': '2px 2px 5px rgba(0,0,0,0.1)',
                'margin-top': '10px'
            }),

            # Box treating the images and the characteristics choices
            html.Div([
                
                # Box to display images and navigation buttons
                html.Div([

                    # Box to display images
                    html.Div([
                        html.Div(id='visualization-box', 
                                style={
                                    'display': 'grid',
                                    'gridTemplateColumns': 'repeat(5, 2fr)',
                                    'gridTemplateRows': 'repeat(2, auto)',
                                    'gap': '5px',
                                    'height': 'auto',
                                    'overflow': 'hidden',
                                    'border': f'2px solid {theme_styles["color"]}',
                                    'border-radius': '20px',
                                    'background-color': theme_styles['background-color'],
                                    'box-shadow': '2px 2px 5px rgba(0,0,0,0.1)'
                                    }
                                ),
                    ], style={'width': '100%', 'margin': '10px', 'overflow': 'auto'}),

                    # Box to display buttons
                    html.Div([
                        html.Button("Previous", 
                                    id='prev-button', 
                                    n_clicks=0, 
                                    style={
                                        'color': theme_styles['text-color'],
                                        'background-color': theme_styles['box-color'],
                                        'padding': '10px 20px',
                                        'margin-right': 'auto',  # Push button to the left
                                        'border-radius': '5px',
                                        'cursor': 'pointer',
                                    }
                        ),

                        html.Button("Next", 
                                    id='next-button', 
                                    n_clicks=0, 
                                    style={
                                        'color': theme_styles['text-color'],
                                        'background-color': theme_styles['box-color'],
                                        'padding': '10px 20px',
                                        'margin-left': 'auto',  # Push button to the right
                                        'border-radius': '5px',
                                        'cursor': 'pointer',
                                    }
                        ),
                    ], style={
                        'display': 'flex', 
                        'justify-content': 'space-between', 
                        'align-items': 'center', 
                        'width': '100%', 
                        'margin': '10px 0'
                    })
                ], style={'width': '60%', 'float': 'left', 'margin': '10px'}),

                # Box to display the characteristics choices and information
                html.Div([

                    # Box to display the people count and the feature count
                    html.Div([
                        html.P(id='people-count', children="People Found: 0", style={'font-weight': 'bold', 'margin': '0'}),
                        html.P(id='feature-count', children="Number of Sorting Features: 0", style={'font-weight': 'bold', 'margin': '0'})
                    ], style={
                        'width': '60%',  # Consistent width
                        'height': 'auto',  # Adjusts height dynamically
                        'margin': '10px auto',  # Centers the div horizontally
                        'display': 'flex',  # Activates flexbox
                        'flex-direction': 'column',  # Stacks children vertically
                        'align-items': 'center',  # Centers content horizontally
                        'justify-content': 'center',  # Centers content vertically
                        'text-align': 'center',  # Aligns text inside the div
                        'background-color': theme_styles['boxes-selected-color'],
                        'border-radius': '30px',
                        'padding': '10px',  # Adds some spacing inside the box
                    }),

                    # Box to display the characteristics choices
                    html.Div([
                        
                        # Dropdown to select characteristics
                        html.Div([
                            dcc.Dropdown(
                                id='characteristics-dropdown',
                                options=[],
                                multi=True,
                                placeholder="Select Characteristics to be Filtered by...",
                                style={
                                    'background-color': theme_styles['background-color'],
                                    'color': theme_styles['color'],
                                    'border-radius': '30px',
                                    'margin-bottom': '10px',
                                }
                            ),
                        ]),

                        # Container to display selected characteristics
                        html.Div(id='selected-characteristics-container', style={
                            'width': '60%',  # Consistent width with the gray box
                            'height': '250px',  # Fixed height to prevent resizing
                            'max-height': '250px',  # Ensures scrolling instead of expansion
                            'overflow': 'auto',  # Adds scrolling when content exceeds height
                            'margin': '10px auto',  # Centers the container horizontally
                            'background-color': theme_styles['background-color'],
                            'border-radius': '20px',
                        })

                    ], style={
                        'margin': 'auto',  # Centers the entire box horizontally
                        'width': '60%',  # Matches the gray box width for consistency
                        'height': '420px',  # Adjusts height dynamically
                        'border-radius': '20px',
                        'box-shadow': '2px 2px 5px rgba(0,0,0,0.1)', 
                        'background-color': theme_styles['boxes-color'],
                    }),

                ], style={
                    'width': '35%',  # Overall container width remains 35%
                    'float': 'right', 
                    'margin': '10px',
                    'height': '100%',
                })

            ], style={  'height': '650px',
                        'width': '100%',
                        'display': 'flex', 
                        'justify-content': 'center', 
                        'background-color': theme_styles['boxes-color'], 
                        'border-radius': '10px', 
                        'box-shadow': '2px 2px 5px rgba(0,0,0,0.1)', 
                        'margin-top': '25px', 
                        'overflow': 'hidden',}),

            # Box to display the average image and the bar plot
            html.Div([

                # Box for average image and generator button
                html.Div([
                    # Container for the average image and loading spinner
                    dcc.Loading(
                        id="loading-average-image",
                        type="default",

                        # Default spinner
                        children=html.Div(id='average-image-container', style={
                            'width': '250px',  # Restrict width to ensure centering
                            'height': '250px',
                            'display': 'flex',  # Use flexbox for centering
                            'align-items': 'center',  # Center content vertically
                            'justify-content': 'center',  # Center content horizontally
                            'border-radius': '20px',
                        })
                    ),

                    # Generate Average Picture button
                    html.Button("Generate Average Picture", id='generate-average-btn', n_clicks=0, style={
                        'cursor': 'pointer',
                        'margin-top': '10px',
                        'background-color': theme_styles['box-color'],
                        'color': theme_styles['text-color'],
                        'padding': '10px 20px',
                        'border-radius': '10px',
                        'text-align': 'center',
                    }),

                ], style={
                    'width': '20%',
                    'height': '100%',  # Ensures it spans vertically
                    'margin': '10px',
                    'display': 'flex',  # Flexbox for alignment
                    'flex-direction': 'column',  # Stack image and button vertically
                    'align-items': 'center',  # Center horizontally
                    'justify-content': 'center',  # Center vertically relative to bar plot
                }),

                # Box for the bar plot and navigation buttons
                html.Div([

                    # Container for the bar plot
                    html.Div(id='attribute-distribution-graph', style={
                        'width': '100%',
                        'height': 'auto',
                        'border': f'2px solid {theme_styles["color"]}'
                    }),

                    # # Box to display buttons to navigate the bar plot
                    # html.Div([

                    #     # Back button
                    #     html.Button("< Back", 
                    #                 id='bar-plot-back-btn', 
                    #                 n_clicks=0, 
                    #                 style={
                    #                     'float': 'left',
                    #                     'color': theme_styles['text-color'],
                    #                     'margin-top': '10px',
                    #                     'cursor': 'pointer',
                    #                     'background-color': theme_styles['box-color'],
                    #                     }
                    #                 ),

                    #     # Next button
                    #     html.Button("Forward >", 
                    #                 id='bar-plot-forward-btn', 
                    #                 n_clicks=0, 
                    #                 style={
                    #                     'float': 'right',
                    #                     'color': theme_styles['text-color'],
                    #                     'margin-top': '10px',
                    #                     'cursor': 'pointer',
                    #                     'background-color': theme_styles['box-color'],
                    #                     }
                    #                 ),

                    # ], style={'clear': 'both', 'margin': '10px 10px'}),

                ], style={
                    'width': '75%',
                    'float': 'right',
                    'margin': '10px',
                    'display': 'flex',
                    'flex-direction': 'column',
                    'justify-content': 'center',  # Align with the average image vertically
                }),

            ], style={
                'height': '450px',
                'margin': '10px',
                'display': 'flex',
                'align-items': 'center',  # Align average image and bar plot vertically
            })

        ], style={'background-color': theme_styles['background-color'], 'height': '100%', 'padding': '10px'})


def register_page_1_callbacks(app, df_s_pg1, df_l_pg1):
    @app.callback(
        Output('characteristics-dropdown', 'options'),
        Input('dataset-store', 'data')
    )
    def update_characteristics_options(selected_dataset):
        """
        Populate the characteristics dropdown based on the selected dataset.
        """
        # Choose the appropriate dataset
        if selected_dataset == 'small':
            df = df_s_pg1
        elif selected_dataset == 'large':
            df = df_l_pg1
        else:
            raise PreventUpdate  # No valid dataset selected

        # Extract characteristic columns (assumes characteristics start from the second column)
        characteristic_columns = df.columns[1:]  # Adjust the index if characteristics don't start from the second column
        return [{'label': col, 'value': col} for col in characteristic_columns]

    @app.callback(
        [Output('attribute-distribution-graph', 'children'),
        Output('selected-characteristics-container', 'children'),
        Output('feature-count', 'children')],
        [Input('dataset-store', 'data'),
        Input('characteristics-dropdown', 'value'),
        Input('attribute-distribution-graph', 'clickData')],
        [State({'type': 'filter-mode', 'index': ALL}, 'value')]
    )
    def update_filtered_bar_plot(dataset, selected_characteristics, click_data, filter_modes):
        # Ensure `selected_characteristics` is not None
        selected_characteristics = selected_characteristics or []

        # Select the dataset
        df = df_s_pg1 if dataset == 'small' else df_l_pg1

        # Apply filters to the dataset
        if selected_characteristics and filter_modes:
            for characteristic, mode in zip(selected_characteristics, filter_modes):
                df = df[df[characteristic] == 1] if mode == 'include' else df[df[characteristic] != 1]

        # Total number of people in the current filtered dataset
        num_selected_people = len(df)

        # Calculate counts for each characteristic in the filtered dataset
        characteristic_counts = df.iloc[:, 1:].sum()  # Exclude the column with image names
        characteristic_counts = characteristic_counts.reset_index()
        characteristic_counts.columns = ['Characteristic', 'Count']

        # Identify selected characteristics
        selected_set = set(selected_characteristics)
        characteristic_counts['Color'] = ['red' if char in selected_set else 'blue' for char in characteristic_counts['Characteristic']]

        # Create the bar plot
        fig = px.bar(
            characteristic_counts,
            x='Characteristic',
            y='Count',
            color='Color',
            color_discrete_map={'red': 'red', 'blue': 'blue'},
            title=f"Characteristic Distribution (Filtered: {num_selected_people} People)"
        )

        # Add a horizontal line at the maximum value (total people in the filtered dataset)
        fig.add_hline(y=num_selected_people, line_dash="dot", line_color="green",
                    annotation_text=f"Max: {num_selected_people}", annotation_position="top right")

        fig.update_layout(
            clickmode='event+select',
            yaxis=dict(range=[0, num_selected_people]),  # Adjust y-axis scale to match selection
            margin=dict(l=20, r=20, t=40, b=40)
        )

        # Update the selected characteristics container
        children = []
        for characteristic in selected_characteristics:
            children.append(html.Div([
                html.Label(f"{characteristic}:", style={'font-weight': 'bold', 'margin-bottom': '5px', 'text-align': 'center'}),
                dcc.RadioItems(
                    id={'type': 'filter-mode', 'index': characteristic},
                    options=[
                        {'label': 'Include', 'value': 'include'},
                        {'label': 'Exclude', 'value': 'exclude'}
                    ],
                    value='include',
                    style={'text-align': 'center'}
                )
            ], style={
                'display': 'flex',
                'flex-direction': 'column',
                'align-items': 'center',
                'margin-bottom': '10px',
            }))

        feature_count = f"Number of Sorting Features: {len(selected_characteristics)}"

        return dcc.Graph(figure=fig), children, feature_count

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
        # Ensure selected_characteristics is a list
        selected_characteristics = selected_characteristics or []

        # Select the dataset
        df = df_s_pg1 if dataset == 'small' else df_l_pg1

        # Apply filters to the dataset
        if selected_characteristics and filter_modes:
            for characteristic, mode in zip(selected_characteristics, filter_modes):
                df = df[df[characteristic] == 1] if mode == 'include' else df[df[characteristic] != 1]

        # Get image paths and count
        images = [f"{image_folder}{row.iloc[0]}" for _, row in df.iterrows()]
        image_names = [row.iloc[0] for _, row in df.iterrows()]
        people_count = f"People Found: {len(df)}"

        # Pagination logic
        page_size = 10
        current_page = (next_clicks - prev_clicks) % (max(1, math.ceil(len(images) / page_size)))
        start_idx, end_idx = current_page * page_size, (current_page + 1) * page_size

        # Handle empty dataset
        if not images:
            return [html.P("No images to display.", style={'textAlign': 'center', 'color': 'red'})], people_count

        # Generate image elements with hover tooltips
        image_elements = [
            html.Img(src=img, title=name, style={'width': '100%', 'height': 'auto', 'border-radius': '10px'})
            for img, name in zip(images[start_idx:end_idx], image_names[start_idx:end_idx])
        ]

        return image_elements, people_count

    @app.callback(
        Output('average-image-container', 'children'),
        Input('generate-average-btn', 'n_clicks'),
        State('dataset-store', 'data'),
        State('characteristics-dropdown', 'value'),
        prevent_initial_call=True
    )
    def generate_average_image(n_clicks, dataset, selected_characteristics):
        """
        Generate the average image based on the current dataset and selected characteristics.
        """
        df = df_s_pg1 if dataset == 'small' else df_l_pg1

        # Filter the dataset based on selected characteristics
        if selected_characteristics:
            for characteristic in selected_characteristics:
                df = df[df[characteristic] == 1]

        # List of image paths for the selected subset
        image_paths = [f"{image_folder_average}{img}" for img in df['image_name']]

        try:
            # Generate the average image
            compute_average_image(image_paths, output_path, resize_dim=(256, 256))
            
            # Return the generated image to the container
            return html.Img(src=f'/assets/{os.path.basename(output_path)}', style={'max-width': '100%', 'border-radius': '10px'})
        except ValueError as e:
            # Handle cases where no valid images are found
            return html.P(str(e), style={'color': 'red'})

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
