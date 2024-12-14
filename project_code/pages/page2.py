# pages/page2.py

import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from functions import get_theme_styles

# Constants for clustering and dimensionality reduction options
dim_red_methods = {'PCA': 'pca', 't-SNE': 'tsne'}
clustering_method_select = {'KMeans': 'kmeans', 'DBSCAN': 'dbscan', 'Hierarchical': 'hierarchical'}
order_select = {'DR -> Clustering': 'dr_clustering', 'Clustering -> DR': 'clustering_dr'}

def render_page_2(theme: str, dataset: str, data) -> html.Div:
    """
    Render Page 2 layout with a design similar to Page 1.
    Args:
        theme (str): Selected theme ('light' or 'dark').
        dataset (str): Selected dataset ('small' or 'large').
        data: Preloaded dataset.
    Returns:
        html.Div: Page layout.
    """
    theme_styles = get_theme_styles(theme)

    return html.Div([
        # Title Section
        html.Div([
            html.H1("Page 2: Dimension Reduction and Clustering", style={
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
            # Parameters Box
            html.Div([
                html.Div([
                    html.Label('Dimension Reduction Method', style={'font-weight': 'bold'}),
                    dcc.RadioItems(
                        id='dimension-reduction-dropdown',
                        options=[{'label': name, 'value': method} for name, method in dim_red_methods.items()],
                        value='pca',
                        style={'margin-bottom': '10px'}
                    ),
                ], style={'margin-bottom': '20px'}),

                html.Div([
                    html.Label('Clustering Method', style={'font-weight': 'bold'}),
                    dcc.RadioItems(
                        id='clustering-method',
                        options=[{'label': name, 'value': method} for name, method in clustering_method_select.items()],
                        value='kmeans',
                        style={'margin-bottom': '10px'}
                    ),
                ], style={'margin-bottom': '20px'}),

                html.Div([
                    html.Label('Order of Operations', style={'font-weight': 'bold'}),
                    dcc.RadioItems(
                        id='order-dropdown',
                        options=[{'label': name, 'value': method} for name, method in order_select.items()],
                        value='dr_clustering',
                        style={'margin-bottom': '10px'}
                    ),
                ], style={'margin-bottom': '20px'}),

                # Dynamic Clustering Options
                html.Div(id='kmeans-options', children=[
                    html.Label("Number of Clusters for KMeans:"),
                    dcc.Input(id='kmeans-clusters', type='number', value=3, style={'margin': '10px'})
                ], style={'margin-bottom': '20px'}),

                html.Div(id='dbscan-options', children=[
                    html.Label("DBSCAN Parameters:"),
                    html.Div([
                        html.Label("Epsilon:"),
                        dcc.Input(id='dbscan-eps', type='number', value=0.5, style={'margin': '10px'})
                    ]),
                    html.Div([
                        html.Label("Min Samples:"),
                        dcc.Input(id='dbscan-min-samples', type='number', value=5, style={'margin': '10px'})
                    ])
                ], style={'display': 'none', 'margin-bottom': '20px'}),

                html.Div(id='hierarchical-options', children=[
                    html.Label("Number of Clusters for Hierarchical Clustering:"),
                    dcc.Input(id='hierarchical-clusters', type='number', value=3, style={'margin': '10px'})
                ], style={'display': 'none', 'margin-bottom': '20px'}),

                # Button to trigger computation
                html.Button("Recalculate/Load Results", id='recalculate', n_clicks=0, style={
                    'background-color': theme_styles['boxes-color'],
                    'color': theme_styles['text-color'],
                    'padding': '10px 20px',
                    'border-radius': '10px',
                    'text-align': 'center',
                    'cursor': 'pointer'
                }),
            ], style={
                'width': '35%',
                'height': '100%',
                'float': 'left',
                'margin': '10px',
                'border-radius': '20px',
                'box-shadow': '2px 2px 5px rgba(0,0,0,0.1)',
                'background-color': theme_styles['boxes-color'],
                'padding': '20px'
            }),

            # Visualization Section
            html.Div([
                html.Div([
                    dcc.Graph(id='tsne-clustering-plot', style={
                        'height': '100%',  # Use full height of the container
                        'width': '100%',   # Use full width of the container
                        'border': f'2px solid {theme_styles["color"]}',
                    }),
                ], style={
                    'width': '100%',
                    'height': '700px',  # Ensure the parent container has a fixed height
                    'border-radius': '20px',
                    'overflow': 'hidden',  # Prevent content overflow
                    'box-shadow': '2px 2px 5px rgba(0,0,0,0.1)',
                    'background-color': theme_styles['boxes-color'],
                }),

                html.Div(id='output-message', style={
                    'text-align': 'center',
                    'margin-top': '20px',
                    'color': theme_styles['color']
                })
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

def register_page_2_callbacks(app):
    @app.callback(
        [Output('kmeans-options', 'style'),
         Output('dbscan-options', 'style'),
         Output('hierarchical-options', 'style')],
        Input('clustering-method', 'value')
    )
    def toggle_clustering_options(clustering_method):
        if clustering_method == 'kmeans':
            return {'margin': '10px'}, {'display': 'none'}, {'display': 'none'}
        elif clustering_method == 'dbscan':
            return {'display': 'none'}, {'margin': '10px'}, {'display': 'none'}
        elif clustering_method == 'hierarchical':
            return {'display': 'none'}, {'display': 'none'}, {'margin': '10px'}
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

    @app.callback(
        [Output('output-message', 'children'),
        Output('tsne-clustering-plot', 'figure')],
        Input('recalculate', 'n_clicks'),
        [State('dataset-store', 'data'),
        State('dimension-reduction-dropdown', 'value'),
        State('clustering-method', 'value'),
        State('kmeans-clusters', 'value'),
        State('dbscan-eps', 'value'),
        State('dbscan-min-samples', 'value'),
        State('hierarchical-clusters', 'value'),
        State('order-dropdown', 'value')]
    )
    def process_clustering(n_clicks, dataset, dim_red, clustering_method, k_clusters, dbscan_eps, dbscan_min_samples, hierarchical_clusters, order):
        if n_clicks == 0:
            return dash.no_update, dash.no_update

        try:
            named = 's' if dataset == 'small' else 'l'
            df = pd.read_csv(f"celeba/celeba_buffalo_{named}.csv")
            features = df.iloc[:, 1:40]

            print("Dataset loaded successfully:")
            print(df.head())

            # Dimensionality Reduction
            if order == 'dr_clustering':
                if dim_red == 'pca':
                    reducer = PCA(n_components=2)
                elif dim_red == 'tsne':
                    reducer = TSNE(n_components=2)
                else:
                    return "Invalid dimension reduction method.", dash.no_update

                reduced_data = reducer.fit_transform(features)
                print("Dimensionality reduction completed:")
                print(reduced_data[:5])

                df['Dim 1'], df['Dim 2'] = reduced_data[:, 0], reduced_data[:, 1]

                if clustering_method == 'kmeans':
                    model = KMeans(n_clusters=k_clusters)
                elif clustering_method == 'dbscan':
                    model = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
                elif clustering_method == 'hierarchical':
                    model = AgglomerativeClustering(n_clusters=hierarchical_clusters)
                else:
                    return "Invalid clustering method.", dash.no_update

                df['Cluster'] = model.fit_predict(reduced_data)
                print("Clustering completed:")
                print(df['Cluster'].unique())

            elif order == 'clustering_dr':
                if clustering_method == 'kmeans':
                    model = KMeans(n_clusters=k_clusters)
                elif clustering_method == 'dbscan':
                    model = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
                elif clustering_method == 'hierarchical':
                    model = AgglomerativeClustering(n_clusters=hierarchical_clusters)
                else:
                    return "Invalid clustering method.", dash.no_update

                df['Cluster'] = model.fit_predict(features)
                print("Clustering completed:")
                print(df['Cluster'].unique())

                if dim_red == 'pca':
                    reducer = PCA(n_components=2)
                elif dim_red == 'tsne':
                    reducer = TSNE(n_components=2)
                else:
                    return "Invalid dimension reduction method.", dash.no_update

                reduced_data = reducer.fit_transform(features)
                print("Dimensionality reduction completed:")
                print(reduced_data[:5])

                df['Dim 1'], df['Dim 2'] = reduced_data[:, 0], reduced_data[:, 1]

            # Generate Scatter Plot
            fig = px.scatter(
                df,
                x='Dim 1',
                y='Dim 2',
                color=df['Cluster'].astype(str),
                title="Clustering Results"
            )
            fig.update_layout(
                autosize=True,  # Automatically adjust size to container
                margin=dict(l=20, r=20, t=40, b=40),  # Adjust margins for better fit
                xaxis=dict(scaleanchor="y", title="Dim 1"),
                yaxis=dict(title="Dim 2")
            )
            return "Clustering completed successfully.", fig
        except Exception as e:
            print("Error occurred:", str(e))
            return f"Error: {e}", dash.no_update

