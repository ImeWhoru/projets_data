import pandas as pd
import numpy as np
from dash import dcc, html, Input, Output
import plotly.express as px
from sklearn.feature_selection import mutual_info_classif

def render_page_3(theme: str, dataset: str, data) -> html.Div:
    """
    Render the layout for Page 3.

    Args:
        theme (str): Current theme ('light' or 'dark').
        dataset (str): Selected dataset ('small' or 'large').
        data: The dataset as a DataFrame.
    """
    return html.Div([
        html.H1("Page 3: Mutual Information Matrix", style={'text-align': 'center'}),
        
        # Section for mutual information matrix
        html.Div([
            html.H2("Mutual Information Matrix"),
            dcc.Graph(id='mutual-info-matrix'),
        ], style={'margin': '20px'}),

        # Section for parameter insights
        html.Div([
            html.H2("Parameter Insights"),
            html.Div([
                html.Div([
                    html.H4("Most Related Parameters"),
                    html.Ul(id='related-parameters'),
                ], style={'width': '45%', 'display': 'inline-block', 'vertical-align': 'top'}),
                html.Div([
                    html.H4("Most Polarizing Parameters"),
                    html.Ul(id='polarizing-parameters'),
                ], style={'width': '45%', 'display': 'inline-block', 'vertical-align': 'top'}),
            ], style={'display': 'flex', 'justify-content': 'space-between'}),
        ], style={'margin': '20px'}),
    ], style={'padding': '20px'})

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
        fig.update_layout(height=600, width=800)

        # Identify most related parameters
        correlations = mutual_info_matrix.abs().mean(axis=0)
        most_related = correlations.nlargest(5).index.tolist()
        related_children = [html.Li(param) for param in most_related]

        # Identify most polarizing parameters
        polarizing_parameters = numeric_data.var().nlargest(5).index.tolist()
        polarizing_children = [html.Li(param) for param in polarizing_parameters]

        return fig, related_children, polarizing_children
