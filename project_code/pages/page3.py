import pandas as pd
import numpy as np
from dash import dcc, html, Input, Output
import plotly.express as px
from sklearn.feature_selection import mutual_info_classif
from functions import get_theme_styles

def render_page_3(theme: str, dataset: str, data) -> html.Div:
    """
    Render Page 3 layout with unified design.

    Args:
        theme (str): Current theme ('light' or 'dark').
        dataset (str): Selected dataset ('small' or 'large').
        data: The dataset as a DataFrame.
    """
    theme_styles = get_theme_styles(theme)

    return html.Div([
        # Title Section
        html.Div([
            html.H1("Page 3: Mutual Information Matrix", style={
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
            'box-shadow': '2px 2px 5px rgba(0,0,0,0.1)',
            'margin-top': '10px'
        }),

        # Main Section
        html.Div([
            # Insights Section
            html.Div([
                html.H2("Parameter Insights", style={
                    'text-align': 'center',
                    'color': theme_styles['text-color'],
                    'margin-bottom': '20px'
                }),
                html.Div([
                    html.Div([
                        html.H4("Most Related Parameters", style={'text-align': 'center'}),
                        html.Ul(id='related-parameters', style={
                            'padding': '10px',
                            'background-color': theme_styles['boxes-color'],
                            'border-radius': '10px',
                            'box-shadow': '2px 2px 5px rgba(0,0,0,0.1)',
                            'margin': '10px',
                            'list-style-type': 'none',
                        }),
                    ], style={'width': '45%', 'margin': '10px'}),
                    html.Div([
                        html.H4("Most Polarizing Parameters", style={'text-align': 'center'}),
                        html.Ul(id='polarizing-parameters', style={
                            'padding': '10px',
                            'background-color': theme_styles['boxes-color'],
                            'border-radius': '10px',
                            'box-shadow': '2px 2px 5px rgba(0,0,0,0.1)',
                            'margin': '10px',
                            'list-style-type': 'none',
                        }),
                    ], style={'width': '45%', 'margin': '10px'}),
                ], style={
                    'display': 'flex',
                    'justify-content': 'space-between',
                    'align-items': 'center',
                }),
            ], style={
                'width': '35%',
                'float': 'left',
                'margin': '10px',
                'border-radius': '20px',
                'box-shadow': '2px 2px 5px rgba(0,0,0,0.1)',
                'background-color': theme_styles['boxes-color'],
                'padding': '20px',
                'height': '700px'
            }),

            # Graph Section
            html.Div([
                dcc.Graph(id='mutual-info-matrix', style={
                    'height': '100%',
                    'width': '100%',
                    'border-radius': '10px',
                    'box-shadow': '2px 2px 5px rgba(0,0,0,0.1)',
                    'background-color': theme_styles['boxes-color']
                }),
            ], style={
                'width': '60%',
                'float': 'right',
                'margin': '10px',
                'border-radius': '20px',
                'box-shadow': '2px 2px 5px rgba(0,0,0,0.1)',
                'background-color': theme_styles['boxes-color'],
                'padding': '20px',
                'height': '700px'
            }),
        ], style={
            'display': 'flex',
            'justify-content': 'space-between',
            'align-items': 'flex-start',
            'margin-top': '25px'
        })
    ], style={'background-color': theme_styles['background-color'], 'padding': '20px'})


def register_page_3_callbacks(app, df_s_pg1, df_l_pg1):
    @app.callback(
        [Output('mutual-info-matrix', 'figure'),
         Output('related-parameters', 'children'),
         Output('polarizing-parameters', 'children')],
        Input('dataset-store', 'data')
    )
    def update_page_3(dataset):
        """
        Generate the mutual information matrix and insights for the dataset.

        Args:
            dataset (str): Selected dataset ('small' or 'large').
        """
        # Select the appropriate dataset
        df = df_s_pg1 if dataset == 'small' else df_l_pg1

        # Extract numeric features only
        numeric_data = df.iloc[:, 1:]  # Exclude the first column (image names)
        
        # Compute Mutual Information
        mutual_info = mutual_info_classif(
            numeric_data, 
            numeric_data.iloc[:, 0],  # Use the first column as the target
            discrete_features=True
        )
        mutual_info_matrix = pd.DataFrame(
            np.corrcoef(numeric_data.T),
            index=numeric_data.columns,
            columns=numeric_data.columns
        )
        
        # Create the heatmap for the mutual information matrix
        fig = px.imshow(
            mutual_info_matrix,
            x=numeric_data.columns,
            y=numeric_data.columns,
            color_continuous_scale='Viridis',
            title="Mutual Information Matrix"
        )
        fig.update_layout(
            autosize=True,
            margin=dict(l=20, r=20, t=40, b=40),
        )

        # Identify most related parameters
        correlations = mutual_info_matrix.abs().mean(axis=0)
        most_related = correlations.nlargest(5).index.tolist()
        related_children = [html.Li(param) for param in most_related]

        # Identify most polarizing parameters
        polarizing_parameters = numeric_data.var().nlargest(5).index.tolist()
        polarizing_children = [html.Li(param) for param in polarizing_parameters]

        return fig, related_children, polarizing_children
