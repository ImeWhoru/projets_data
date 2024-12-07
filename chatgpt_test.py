import dash
from dash import dcc, html, Input, Output, ctx
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
import os

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "t-SNE with Clustering Methods"

# Define the app layout
app.layout = html.Div([
    html.H1("t-SNE with Clustering Methods", style={'text-align': 'center'}),
    html.Div([
        html.Button("Load celeba_buffalo_s.csv", id='load-s', n_clicks=0, className='button'),
        html.Button("Load celeba_buffalo_l.csv", id='load-l', n_clicks=0, className='button'),
    ], style={'text-align': 'center', 'margin': '10px'}),
    
    html.Div([
        html.Label("Choose Clustering Method:"),
        dcc.RadioItems(
            id='clustering-method',
            options=[
                {'label': 'KMeans', 'value': 'kmeans'},
                {'label': 'DBSCAN', 'value': 'dbscan'},
                {'label': 'Hierarchical', 'value': 'hierarchical'}
            ],
            value='dbscan',
            inline=True
        )
    ], style={'text-align': 'center', 'margin': '10px'}),

    html.Div([
        html.Button("Recalculate Results", id='recalculate', n_clicks=0, className='button'),
        html.Button("Load Results", id='load-results', n_clicks=0, className='button'),
    ], style={'text-align': 'center', 'margin': '10px'}),

    dcc.Graph(id='tsne-clustering-plot'),
    html.Div(id='output-message', style={'text-align': 'center', 'margin': '10px'})
])

# Callback to handle button clicks, perform t-SNE, and apply the selected clustering method
@app.callback(
    [Output('output-message', 'children'),
     Output('tsne-clustering-plot', 'figure')],
    [Input('load-s', 'n_clicks'),
     Input('load-l', 'n_clicks'),
     Input('clustering-method', 'value'),
     Input('recalculate', 'n_clicks'),
     Input('load-results', 'n_clicks')]
)
def process_file(load_s_clicks, load_l_clicks, clustering_method, recalculate_clicks, load_results_clicks):
    global previous_results

    # Determine which button was clicked
    ctx_triggered = ctx.triggered_id
    if ctx_triggered is None:
        raise dash.exceptions.PreventUpdate

    # If "Load Results" button is clicked, load previously saved results
    if ctx_triggered == 'load-results':
        if previous_results:
            return previous_results['message'], previous_results['figure']
        else:
            return "No results found to load. Please recalculate first.", dash.no_update

    # If "Recalculate Results" button is clicked, or a file button is clicked, compute or load from file
    if ctx_triggered == 'recalculate' or ctx_triggered in ['load-s', 'load-l']:
        # Select the file based on the clicked button
        if ctx_triggered == 'load-s':
            filename = 'celeba/celeba_buffalo_s.csv'
        elif ctx_triggered == 'load-l':
            filename = 'celeba/celeba_buffalo_l.csv'
        else:
            raise dash.exceptions.PreventUpdate

        # Define the result filename
        result_filename = filename.replace('.csv', f'_{clustering_method}_results.csv')

        # Check if results already exist
        if os.path.exists(result_filename):
            # If results file exists, load it
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
            # Load the CSV file
            try:
                df = pd.read_csv(filename)
            except Exception as e:
                return f"Error loading file: {e}", dash.no_update

            # Validate the file structure
            required_columns = ['image_name', 'id']
            if not all(col in df.columns for col in required_columns):
                return "Invalid CSV file format. Ensure it contains 'image_name', 'id', features, and embeddings.", dash.no_update

            # Extract features and embeddings
            features = df.iloc[:, 1:40]  # Assuming features are from index 1 to 39
            embeddings = df.iloc[:, 40:-1]  # Assuming embeddings are from index 40 to the second last column
            data = pd.concat([features, embeddings], axis=1)

            # Perform t-SNE dimensionality reduction
            tsne = TSNE(n_components=2, random_state=42)
            tsne_results = tsne.fit_transform(data)

            # Add t-SNE results to DataFrame
            df['t-SNE 1'] = tsne_results[:, 0]
            df['t-SNE 2'] = tsne_results[:, 1]

            # Apply the selected clustering method
            if clustering_method == 'dbscan':
                cluster_model = DBSCAN(eps=5, min_samples=10)  # Tune parameters as needed
            elif clustering_method == 'kmeans':
                cluster_model = KMeans(n_clusters=5, random_state=42)  # Adjust number of clusters
            elif clustering_method == 'hierarchical':
                cluster_model = AgglomerativeClustering(n_clusters=5)  # Adjust number of clusters
            else:
                return "Invalid clustering method selected.", dash.no_update

            clusters = cluster_model.fit_predict(tsne_results)
            df['Cluster'] = clusters

            # Save results to a CSV file
            df.to_csv(result_filename, index=False)

            # Generate scatter plot
            fig = px.scatter(
                df,
                x='t-SNE 1',
                y='t-SNE 2',
                color=df['Cluster'].astype(str),
                hover_name='image_name',
                title=f"t-SNE Results with {clustering_method.upper()} Clustering ({filename})",
                labels={'Cluster': 'Cluster ID'}
            )

            # Save results globally for future use
            previous_results = {
                'message': f"File '{filename}' processed successfully with {clustering_method.upper()} applied and results saved.",
                'figure': fig
            }

            return previous_results['message'], previous_results['figure']


# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
