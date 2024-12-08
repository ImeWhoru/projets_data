import math
from dash import html, dcc, Input, Output, State, ctx
from dash.dependencies import ALL
from functions import get_theme_styles

# Define the folder for images
image_folder = 'assets/img_celeba/'  # Dash serves from `assets/`

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
        """
        Populate the characteristics dropdown options based on the selected dataset.
        """
        df = df_s_pg1 if selected_dataset == 'small' else df_l_pg1

        # Verify and exclude non-characteristic columns (like IDs or paths)
        characteristic_columns = df.columns[1:]  # Adjust this as per your dataset

        return [{'label': col, 'value': col} for col in characteristic_columns]

    @app.callback(
        [Output('selected-characteristics-container', 'children'),
        Output('feature-count', 'children')],
        Input('characteristics-dropdown', 'value')
    )
    def update_selected_characteristics(selected_characteristics):
        """
        Update the list of selected characteristics with Include/Exclude radio items in a single line.
        """
        if not selected_characteristics:
            # No characteristics selected; return a message and zero count
            return html.P("No characteristics selected.", style={'color': 'gray', 'textAlign': 'center'}), "Number of Features Selected: 0"

        # Generate a row for each selected characteristic with Include/Exclude options
        children = []
        for characteristic in selected_characteristics:
            children.append(html.Div([
                html.Label(f"{characteristic}:", style={'font-weight': 'bold', 'margin-right': '10px', 'display': 'inline-block'}),
                dcc.RadioItems(
                    id={'type': 'filter-mode', 'index': characteristic},
                    options=[
                        {'label': 'Include', 'value': 'include'},
                        {'label': 'Exclude', 'value': 'exclude'}
                    ],
                    value='include',
                    style={'display': 'inline-block', 'margin-left': '10px'}
                )
            ], style={'margin-bottom': '10px', 'display': 'flex', 'align-items': 'center'}))  # Flex layout for alignment

        # Feature count text
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
        """
        Update the visualization and people count based on selected characteristics and pagination.
        """
        df = df_s_pg1 if dataset == 'small' else df_l_pg1

        # Apply filtering logic
        if selected_characteristics and filter_modes:
            for characteristic, mode in zip(selected_characteristics, filter_modes):
                if mode == 'include':
                    df = df[df[characteristic] == 1]
                elif mode == 'exclude':
                    df = df[df[characteristic] != 1]

        # Generate the list of image paths
        images = [f"{image_folder}{row.iloc[0]}" for _, row in df.iterrows()]
        people_count = f"People Found: {len(df)}"
        page_size = 10
        total_pages = max(1, math.ceil(len(images) / page_size))

        # Corrected logic for page calculation
        current_page = (next_clicks - prev_clicks) % total_pages if len(images) > 0 else 0
        start_idx, end_idx = current_page * page_size, min(len(images), (current_page + 1) * page_size)

        # Handle no images
        if len(images) == 0:
            return [html.P("No images to display.", style={'textAlign': 'center', 'color': 'red'})], people_count

        # Display images for the current page with tooltips
        return [
            html.Img(
                src=img,
                title=img.split("/")[-1],  # Extract the image name from the path
                style={'width': '100%', 'padding': '5px'}
            ) for img in images[start_idx:end_idx]
        ], people_count
