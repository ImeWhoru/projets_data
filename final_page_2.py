import dash
from dash import dcc, html, Input, Output, ctx, State
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering

dataset_select = {'Small Dataset': 'celeba/celeba_buffalo_s.csv', 'Large Dataset': 'celeba/celeba_buffalo_l.csv'}
dim_red_methods = {'PCA': 'pca', 't-SNE': 'tsne'}
clustering_method_select = {'KMeans': 'kmeans', 'DBSCAN': 'dbscan', 'Hierarchical': 'hierarchical'}

# Initialization
app = dash.Dash(__name__)
app.title = "Page 2"

app.layout = html.Div([
    html.H1("Page 2: Dimension Reduction and Clustering", style={'font-size': '60px', 'text-align': 'center'}),
    html.Div([html.Label('Please select the wanted dataset'),
              dcc.Dropdown(id='dataset-dropdown',
                           options=[{'label': name, 'value': method} for name, method in dataset_select.items()],
                           value=list(dataset_select.values())[0])],
             style={'font-size': '20px', 'margin': '10px'}),
    html.Div([html.Label('Please select the dimension reduction method'),
              dcc.RadioItems(id='dimension-reduction-dropdown',
                             options=[{'label': name, 'value': method} for name, method in dim_red_methods.items()],
                             value='pca')],
             style={'font-size': '20px', 'margin': '10px'}),
    html.Div([html.Label('Please select the clustering method'),
              dcc.RadioItems(id='clustering-method',
                             options=[{'label': name, 'value': method} for name, method in clustering_method_select.items()],
                             value='kmeans')],
             style={'font-size': '20px', 'margin': '10px'}),
    html.Div(id='kmeans-options', children=[
        html.Label("Number of Clusters for KMeans:"),
        dcc.Input(id='kmeans-clusters', type='number', value=3, style={'margin': '10px'})
    ], style={'margin': '10px'}),
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
    ], style={'display': 'none', 'margin': '10px'}),
    html.Div(id='hierarchical-options', children=[
        html.Label("Number of Clusters for Hierarchical Clustering:"),
        dcc.Input(id='hierarchical-clusters', type='number', value=3, style={'margin': '10px'})
    ], style={'display': 'none', 'margin': '10px'}),
    html.Button("Recalculate/Load Results", id='recalculate', n_clicks=0, style={'margin': '10px'}),
    dcc.Loading(id="loading-spinner", children=[
        html.Div(id='output-message'),
        html.Div([dcc.Graph(id='tsne-clustering-plot', style={'height': '700px'})],     style={
        'display': 'flex',
        'justify-content': 'center',
        'align-items': 'center'
        })
    ])
])

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
    [State('dataset-dropdown', 'value'),
     State('dimension-reduction-dropdown', 'value'),
     State('clustering-method', 'value'),
     State('kmeans-clusters', 'value'),
     State('dbscan-eps', 'value'),
     State('dbscan-min-samples', 'value'),
     State('hierarchical-clusters', 'value')]
)
def process_clustering(n_clicks, dataset, dim_red, clustering_method, k_clusters, dbscan_eps, dbscan_min_samples, hierarchical_clusters):
    if n_clicks == 0:
        return dash.no_update, dash.no_update

    try:
        df = pd.read_csv(dataset)
        features = df.iloc[:, 1:40]

        if dim_red == 'pca':
            reducer = PCA(n_components=2)
        elif dim_red == 'tsne':
            reducer = TSNE(n_components=2)
        else:
            return "Invalid dimension reduction method.", dash.no_update

        reduced_data = reducer.fit_transform(features)
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
        fig = px.scatter(
    df,
    x='Dim 1',
    y='Dim 2',
    color=df['Cluster'].astype(str),
    title="Clustering Results"
)

# Update the layout to enforce a square plot
        fig.update_layout(
    width=700,  # Set a square width
    height=700,  # Set a square height
    xaxis=dict(scaleanchor="y",  # Link x-axis scale to y-axis
               title="Dim 1"),
    yaxis=dict(title="Dim 2")
)


        return "Clustering completed successfully.", fig
    except Exception as e:
        return f"Error: {e}", dash.no_update

if __name__ == '__main__':
    app.run_server(debug=True)
