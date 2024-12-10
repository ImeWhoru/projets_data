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
            placeholder="Select a dataset",
        )
    ], style={'margin': '10px'}),
    
    html.Div([
        html.Label("Choose Clustering Method:"),
        dcc.RadioItems(
            id='clustering-method',
            options=[{'label': name, 'value': method} for name, method in clustering_methods.items()],
            value='dbscan',
            inline=True
        )
    ], style={'margin': '10px'}),
    
    html.Div([
        html.Label("KMeans Clusters:"),
        dcc.Input(id='kmeans-clusters', type='number', min=1, value=5, step=1),
    ], id='kmeans-options', style={'display': 'none', 'margin': '10px'}),
    
    html.Div([
        html.Label("DBSCAN Parameters:"),
        dcc.Input(id='dbscan-eps', type='number', min=0.1, step=0.1, value=5, placeholder="eps"),
        dcc.Input(id='dbscan-min-samples', type='number', min=1, value=10, step=1, placeholder="min_samples"),
    ], id='dbscan-options', style={'display': 'none', 'margin': '10px'}),
    
    html.Div([
        html.Label("Hierarchical Clusters:"),
        dcc.Input(id='hierarchical-clusters', type='number', min=1, value=5, step=1),
    ], id='hierarchical-options', style={'display': 'none', 'margin': '10px'}),
    
    html.Div([
        html.Button("Recalculate Results", id='recalculate', n_clicks=0, className='button'),
        html.Button("Load Results", id='load-results', n_clicks=0, className='button'),
    ], style={'text-align': 'center', 'margin': '10px'}),
    
    dcc.ConfirmDialog(
        id='confirm-recalculate',
        message="Are you sure you want to recalculate the results? This may take a long time and overwrite already calculated results.",
    ),
    
    dcc.Loading(
        id="loading-spinner",
        type="default",
        children=[
            dcc.Graph(id='tsne-clustering-plot'),
            html.Div(id='output-message', style={'text-align': 'center', 'margin': '10px'})
        ]
    ),

    html.Div(id='loading-message', style={'text-align': 'center', 'margin': '10px', 'font-weight': 'bold'}),
])

# Callback to toggle parameter inputs based on clustering method
@app.callback(
    [Output('kmeans-options', 'style'),
     Output('dbscan-options', 'style'),
     Output('hierarchical-options', 'style')],
    [Input('clustering-method', 'value')]
)
def toggle_options(clustering_method):
    if clustering_method == 'kmeans':
        return {'margin': '10px'}, {'display': 'none'}, {'display': 'none'}
    elif clustering_method == 'dbscan':
        return {'display': 'none'}, {'margin': '10px'}, {'display': 'none'}
    elif clustering_method == 'hierarchical':
        return {'display': 'none'}, {'display': 'none'}, {'margin': '10px'}
    return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

# Callback for showing the confirmation dialog
@app.callback(
    Output('confirm-recalculate', 'displayed'),
    Input('recalculate', 'n_clicks')
)
def show_confirmation(n_clicks):
    return n_clicks > 0

# Callback for processing data (executed after confirmation)
@app.callback(
    [Output('output-message', 'children'),
     Output('tsne-clustering-plot', 'figure')],
    [Input('confirm-recalculate', 'submit_n_clicks'),
     Input('load-results', 'n_clicks')],
    [State('dataset-dropdown', 'value'),
     State('clustering-method', 'value'),
     State('kmeans-clusters', 'value'),
     State('dbscan-eps', 'value'),
     State('dbscan-min-samples', 'value'),
     State('hierarchical-clusters', 'value')]
)
def process_file(confirm_recalculate_clicks, load_results_clicks, selected_dataset, clustering_method, kmeans_clusters, dbscan_eps, dbscan_min_samples, hierarchical_clusters):
    # Check if either button was clicked
    if ctx.triggered_id == 'load-results' and load_results_clicks > 0:
        # Load precomputed results
        result_filename = selected_dataset.replace('.csv', f'_{clustering_method}_results.csv')
        if os.path.exists(result_filename):
            df = pd.read_csv(result_filename)
            fig = px.scatter(
                df,
                x='t-SNE 1',
                y='t-SNE 2',
                color=df['Cluster'].astype(str),
                hover_name='image_name',
                title=f"t-SNE Results with {clustering_method.upper()} Clustering (Loaded)",
                labels={'Cluster': 'Cluster ID'}
            )
            fig.update_layout(width=700, height=700, xaxis=dict(scaleanchor="y", title="t-SNE 1"), yaxis=dict(title="t-SNE 2"))
            return f"Loaded results from file: {result_filename}", fig
        else:
            return "No precomputed results found. Please recalculate.", dash.no_update

    if ctx.triggered_id == 'confirm-recalculate' and confirm_recalculate_clicks > 0:
        # Recalculate results
        try:
            df = pd.read_csv(selected_dataset)
            features = df.iloc[:, 1:40]
            embeddings = df.iloc[:, 40:-1]
            data = pd.concat([features, embeddings], axis=1)
            tsne = TSNE(n_components=2, random_state=42)
            tsne_results = tsne.fit_transform(data)

            df['t-SNE 1'] = tsne_results[:, 0]
            df['t-SNE 2'] = tsne_results[:, 1]

            if clustering_method == 'kmeans':
                cluster_model = KMeans(n_clusters=kmeans_clusters, random_state=42)
            elif clustering_method == 'dbscan':
                cluster_model = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
            elif clustering_method == 'hierarchical':
                cluster_model = AgglomerativeClustering(n_clusters=hierarchical_clusters)
            else:
                return "Invalid clustering method selected.", dash.no_update

            clusters = cluster_model.fit_predict(tsne_results)
            df['Cluster'] = clusters

            result_filename = selected_dataset.replace('.csv', f'_{clustering_method}_results.csv')
            df.to_csv(result_filename, index=False)

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
            return f"Results recalculated and saved to {result_filename}.", fig
        except Exception as e:
            return f"Error: {e}", dash.no_update

    return dash.no_update, dash.no_update

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
