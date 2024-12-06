import dash
from dash import dcc, html, Input, Output, ctx
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "t-SNE with DBSCAN Clustering"

# Define the app layout
app.layout = html.Div([
    html.H1("t-SNE with DBSCAN Clustering", style={'text-align': 'center'}),
    html.Div([
        html.Button("Load celeba_buffalo_s.csv", id='load-s', n_clicks=0, className='button'),
        html.Button("Load celeba_buffalo_l.csv", id='load-l', n_clicks=0, className='button'),
    ], style={'text-align': 'center', 'margin': '10px'}),
    dcc.Graph(id='tsne-dbscan-plot'),
    html.Div(id='output-message', style={'text-align': 'center', 'margin': '10px'})
])

# Callback to handle button clicks, perform t-SNE, and apply DBSCAN
@app.callback(
    [Output('output-message', 'children'),
     Output('tsne-dbscan-plot', 'figure')],
    [Input('load-s', 'n_clicks'),
     Input('load-l', 'n_clicks')]
)
def process_file(load_s_clicks, load_l_clicks):
    # Determine which button was clicked
    ctx_triggered = ctx.triggered_id
    if ctx_triggered is None:
        raise dash.exceptions.PreventUpdate

    # Select the file based on the clicked button
    if ctx_triggered == 'load-s':
        filename = 'celeba/celeba_buffalo_s.csv'
    elif ctx_triggered == 'load-l':
        filename = 'celeba/celeba_buffalo_l.csv'
    else:
        raise dash.exceptions.PreventUpdate

    # Load the selected file
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

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=10, min_samples=10)  # Tune parameters as needed
    clusters = dbscan.fit_predict(tsne_results)
    df['Cluster'] = clusters

    # Generate scatter plot
    fig = px.scatter(
        df,
        x='t-SNE 1',
        y='t-SNE 2',
        color=df['Cluster'].astype(str),
        hover_name='image_name',
        title=f"t-SNE Results with DBSCAN Clustering ({filename})",
        labels={'Cluster': 'Cluster ID'}
    )

    return f"File '{filename}' processed successfully with DBSCAN applied.", fig


# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
