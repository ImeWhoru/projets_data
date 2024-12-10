import dash
from dash import dcc, html, Input, Output, ctx, State
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
import os

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "t-SNE with Clustering Methods"

# Define options for datasets and clustering methods
datasets = {
    "Small Dataset": "celeba/celeba_buffalo_s.csv",
    "Large Dataset": "celeba/celeba_buffalo_l.csv"
}
clustering_methods = {
    "KMeans": "kmeans",
    "DBSCAN": "dbscan",
    "Hierarchical": "hierarchical"
}

# App layout
app.layout = html.Div([
    html.H1("t-SNE with Clustering Methods", style={'text-align': 'center'}),

    html.Div([
        html.Label("Select Dataset:"),
        dcc.Dropdown(
            id='dataset-dropdown',
            options=[{'label': name, 'value': path} for name, path in datasets.items()],
            value=datasets["Small Dataset"],  # Set default value to the small dataset
            placeholder="Select a dataset",
        )
    ], style={'margin': '10px'}),

    html.Div([
        html.Label("Choose Clustering Method:"),
        dcc.RadioItems(
            id='clustering-method',
            options=[{'label': name, 'value': method} for name, method in clustering_methods.items()],
            value='kmeans',
            inline=True
        )
    ], style={'margin': '10px'}),

    # Conditional display for clustering parameters
    html.Div([
        html.Label("KMeans Clusters:"),
        dcc.Input(id='kmeans-clusters', type='number', min=1, value=5, step=1),
    ], id='kmeans-options', style={'display': 'none', 'margin': '10px'}),

    html.Div([
        html.Label("DBSCAN Parameters:"),
        html.Div([
            html.Label("Epsilon (eps):", style={'margin-right': '10px'}),
            dcc.Input(id='dbscan-eps', type='number', min=0.1, step=0.1, value=5),
        ], style={'margin-bottom': '10px'}),
        html.Div([
            html.Label("Min Samples:", style={'margin-right': '10px'}),
            dcc.Input(id='dbscan-min-samples', type='number', min=1, value=10, step=1),
        ]),
    ], id='dbscan-options', style={'display': 'none', 'margin': '10px'}),

    html.Div([
        html.Label("Hierarchical Clustering Parameter:"),
        html.Div([
            html.Label("Number of Clusters:", style={'margin-right': '10px'}),
            dcc.Input(id='hierarchical-clusters', type='number', min=1, value=5, step=1),
        ], style={'margin-bottom': '10px'}),
    ], id='hierarchical-options', style={'display': 'none', 'margin': '10px'}),

    # Buttons for recalculating or loading results
    html.Div([
        html.Button("Recalculate Results", id='recalculate', n_clicks=0),
        html.Button("Load Results", id='load-results', n_clicks=0),
    ], style={'text-align': 'center', 'margin': '10px'}),

    # Confirmation Dialogs
    dcc.ConfirmDialog(
        id='confirm-recalculate',
        message="Are you sure you want to recalculate the results? This may take a long time and overwrite already calculated results.",
    ),

    dcc.ConfirmDialog(
        id='confirm-load-results',
        message="No precomputed results found. Would you like to recalculate the results now?",
    ),

    # Loading Spinner
    dcc.Loading(
        id="loading-spinner",
        type="default",
        children=[
            html.Div(
                dcc.Graph(
                    id='tsne-clustering-plot',
                    figure={
                        'data': [],
                        'layout': {
                            'xaxis': {'visible': False},
                            'yaxis': {'visible': False},
                            'annotations': [{
                                'text': 'Waiting for instruction',
                                'xref': 'paper',
                                'yref': 'paper',
                                'showarrow': False,
                                'font': {'size': 20}
                            }]
                        }
                    }
                ),
                style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center'}
            ),
            html.Div(id='output-message', style={'text-align': 'center', 'margin': '10px'})
        ]
    ),

    # Loading message display
    html.Div(id='loading-message', style={'text-align': 'center', 'margin': '10px', 'font-weight': 'bold'}),
])

# Callback to toggle parameter inputs based on clustering method
@app.callback(
    [Output('kmeans-options', 'style'),
     Output('dbscan-options', 'style'),
     Output('hierarchical-options', 'style')],
    Input('clustering-method', 'value')
)
def toggle_options(clustering_method):
    if clustering_method == 'kmeans':
        return {'margin': '10px'}, {'display': 'none'}, {'display': 'none'}
    elif clustering_method == 'dbscan':
        return {'display': 'none'}, {'margin': '10px'}, {'display': 'none'}
    elif clustering_method == 'hierarchical':
        return {'display': 'none'}, {'display': 'none'}, {'margin': '10px'}
    return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

# Callback for showing the confirmation dialog for recalculation
@app.callback(
    Output('confirm-recalculate', 'displayed'),
    Input('recalculate', 'n_clicks')
)
def show_confirmation_recalculate(n_clicks):
    return n_clicks > 0

# Callback for showing the confirmation dialog for loading results
@app.callback(
    Output('confirm-load-results', 'displayed'),
    Input('load-results', 'n_clicks')
)
def show_load_confirmation(n_clicks):
    return n_clicks > 0

# Callback for processing data after confirmation dialog submission
@app.callback(
    [Output('output-message', 'children'),
     Output('tsne-clustering-plot', 'figure')],
    [Input('confirm-recalculate', 'submit_n_clicks'),
     Input('confirm-load-results', 'submit_n_clicks')],
    [State('dataset-dropdown', 'value'),
     State('clustering-method', 'value'),
     State('kmeans-clusters', 'value'),
     State('dbscan-eps', 'value'),
     State('dbscan-min-samples', 'value'),
     State('hierarchical-clusters', 'value')]
)
def process_file(confirm_recalculate_clicks, confirm_load_results_clicks, selected_dataset, clustering_method, kmeans_clusters, dbscan_eps, dbscan_min_samples, hierarchical_clusters):
    # Determine which action was triggered
    triggered_id = ctx.triggered_id

    # Loading the dataset and performing t-SNE and clustering
    try:
        df = pd.read_csv(selected_dataset)
        features = df.iloc[:, 1:40]
        embeddings = df.iloc[:, 40:-1]
        data = pd.concat([features, embeddings], axis=1)
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(data)

        df['t-SNE 1'] = tsne_results[:, 0]
        df['t-SNE 2'] = tsne_results[:, 1]

        # Select clustering method based on user input
        if clustering_method == 'kmeans':
            cluster_model = KMeans(n_clusters=kmeans_clusters)
        elif clustering_method == 'dbscan':
            cluster_model = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        elif clustering_method == 'hierarchical':
            cluster_model = AgglomerativeClustering(n_clusters=hierarchical_clusters)
        else:
            return "Invalid clustering method selected.", dash.no_update

        clusters = cluster_model.fit_predict(tsne_results)
        df['Cluster'] = clusters

        # Save results
        result_filename = selected_dataset.replace('.csv', f'_{clustering_method}_results.csv')
        df.to_csv(result_filename, index=False)

        # Generate plot
        fig = px.scatter(
            df,
            x='t-SNE 1',
            y='t-SNE 2',
            color=df['Cluster'].astype(str),
            hover_name='image_name',
            title=f"t-SNE Results with {clustering_method.upper()} Clustering",
            labels={'Cluster': 'Cluster ID'}
        )
        fig.update_layout(width=700, height=700, xaxis=dict(scaleanchor="y", title="t-SNE 1"), yaxis=dict(title="t-SNE 2"))

        # Return message and plot
        if triggered_id == 'confirm-load-results' and confirm_load_results_clicks > 0:
            return f"Results recalculated and saved to {result_filename}.", fig
        return "Recalculating...", dash.no_update
    except Exception as e:
        return f"Error: {e}", dash.no_update

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
