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

# Callback to show the confirmation dialog when "Recalculate" is clicked
@app.callback(
    Output('confirm-recalculate', 'displayed'),
    [Input('recalculate', 'n_clicks')]
)
def show_confirmation(n_clicks):
    if n_clicks > 0:
        return True
    return False

# Callback to display loading message
@app.callback(
    Output('loading-message', 'children'),
    [Input('confirm-recalculate', 'submit_n_clicks'),
     Input('load-results', 'n_clicks')],
    [State('dataset-dropdown', 'value')]
)
def display_loading_message(confirm_recalculate_clicks, load_results_clicks, selected_dataset):
    ctx_triggered = ctx.triggered_id
    if not selected_dataset:
        return "Please select a dataset."
    if ctx_triggered == 'confirm-recalculate' and confirm_recalculate_clicks > 0:
        return "Calculating results..."
    elif ctx_triggered == 'load-results' and load_results_clicks > 0:
        return "Loading results..."
    return ""

# Callback for processing data (executed after confirmation)
@app.callback(
    [Output('output-message', 'children'),
     Output('tsne-clustering-plot', 'figure')],
    [Input('confirm-recalculate', 'submit_n_clicks'),
     Input('load-results', 'n_clicks')],
    [State('dataset-dropdown', 'value'),
     State('clustering-method', 'value')]
)
def process_file(confirm_recalculate_clicks, load_results_clicks, selected_dataset, clustering_method):
    ctx_triggered = ctx.triggered_id
    if ctx_triggered is None or not selected_dataset or not clustering_method:
        return "Please select a dataset and clustering method.", dash.no_update

    result_filename = selected_dataset.replace('.csv', f'_{clustering_method}_results.csv')

    # Load results if "Load Results" is clicked
    if ctx_triggered == 'load-results':
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
            return f"Loaded results from file: {result_filename}", fig
        else:
            return "No precomputed results found. Please recalculate.", dash.no_update

    # Recalculate results if "Recalculate" is confirmed
    if ctx_triggered == 'confirm-recalculate' and confirm_recalculate_clicks > 0:
        try:
            # Load the selected dataset
            df = pd.read_csv(selected_dataset)
        except Exception as e:
            return f"Error loading dataset: {e}", dash.no_update

        # Ensure required columns are present
        if not all(col in df.columns for col in ['image_name', 'id']):
            return "Invalid dataset format. Required columns: 'image_name', 'id', features, and embeddings.", dash.no_update

        # Extract features and perform t-SNE
        features = df.iloc[:, 1:40]
        embeddings = df.iloc[:, 40:-1]
        data = pd.concat([features, embeddings], axis=1)
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(data)

        # Add t-SNE results to the dataframe
        df['t-SNE 1'] = tsne_results[:, 0]
        df['t-SNE 2'] = tsne_results[:, 1]

        # Apply selected clustering method
        if clustering_method == 'dbscan':
            cluster_model = DBSCAN(eps=5, min_samples=10)
        elif clustering_method == 'kmeans':
            cluster_model = KMeans(n_clusters=5, random_state=42)
        elif clustering_method == 'hierarchical':
            cluster_model = AgglomerativeClustering(n_clusters=5)
        else:
            return "Invalid clustering method selected.", dash.no_update

        clusters = cluster_model.fit_predict(tsne_results)
        df['Cluster'] = clusters

        # Save results to file
        df.to_csv(result_filename, index=False)

        # Generate plot
        fig = px.scatter(
            df,
            x='t-SNE 1',
            y='t-SNE 2',
            color=df['Cluster'].astype(str),
            hover_name='image_name',
            title=f"t-SNE Results with {clustering_method.upper()} Clustering ({selected_dataset})",
            labels={'Cluster': 'Cluster ID'}
        )

        return f"Results recalculated and saved to {result_filename}.", fig


# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
