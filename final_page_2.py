import dash
from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate  # Fixed missing import
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering

dataset_select = {'Small Dataset': 'celeba/celeba_buffalo_s.csv', 'Large Dataset': 'celeba/celeba_buffalo_l.csv'}
dim_red_methods = {'PCA': 'pca', 't-SNE': 'tsne'}
clustering_method_select = {'KMeans': 'kmeans', 'DBSCAN': 'dbscan', 'Hierarchical': 'hierarchical'}
order_select = {'DR -> Clustering': 'dr_clustering', 'Clustering -> DR': 'clustering_dr'}

# Define your theme styles (example, adjust as needed)
theme_styles = {
    'background-color': '#f0f0f0',
    'color': '#333',
    'boxes-color': '#ffffff',
    'boxes-selected-color': '#e0e0e0'
}

# Initialization
app = dash.Dash(__name__)
app.title = "Page 2"

app.layout = html.Div([
    html.H1("Page 2: Dimension Reduction and Clustering", style={'font-size': '60px', 'text-align': 'center'}),
    html.Div([
        # Box to display the people count and the feature count
        html.Div([
            html.P(id='people-count', children="People Found: 0", style={'font-weight': 'bold', 'margin': '0'}),
            html.P(id='feature-count', children="Number of Sorting Features: 0", style={'font-weight': 'bold', 'margin': '0'})
        ], style={
            'width': '60%',  # Consistent width
            'height': 'auto',  # Adjusts height dynamically
            'margin': '10px auto',  # Centers the div horizontally
            'display': 'flex',  # Activates flexbox
            'flex-direction': 'column',  # Stacks children vertically
            'align-items': 'center',  # Centers content horizontally
            'justify-content': 'center',  # Centers content vertically
            'text-align': 'center',  # Aligns text inside the div
            'background-color': theme_styles['boxes-selected-color'],
            'border-radius': '30px',
            'padding': '10px',  # Adds some spacing inside the box
        }),

        # Box to display the characteristics choices
        html.Div([

            # Dropdown to select characteristics
            html.Div([
                dcc.Dropdown(
                    id='characteristics-dropdown',
                    options=[],
                    multi=True,
                    placeholder="Select Characteristics to be Filtered by...",
                    style={
                        'background-color': theme_styles['background-color'],
                        'color': theme_styles['color'],
                        'border-radius': '30px',
                        'margin-bottom': '10px',
                    }
                ),
            ]),

            # Container to display selected characteristics
            html.Div(id='selected-characteristics-container', style={
                'width': '60%',  # Consistent width with the gray box
                'height': '250px',  # Fixed height to prevent resizing
                'max-height': '250px',  # Ensures scrolling instead of expansion
                'overflow': 'auto',  # Adds scrolling when content exceeds height
                'margin': '10px auto',  # Centers the container horizontally
                'background-color': theme_styles['background-color'],
                'border-radius': '20px',
            })

        ], style={
            'margin': 'auto',  # Centers the entire box horizontally
            'width': '60%',  # Matches the gray box width for consistency
            'height': '420px',  # Adjusts height dynamically
            'border-radius': '20px',
            'box-shadow': '2px 2px 5px rgba(0,0,0,0.1)',
            'background-color': theme_styles['boxes-color'],
        }),

    ], style={
        'width': '35%',  # Overall container width remains 35%
        'float': 'right',
        'margin': '10px',
        'height': '100%',
    }),

    # Box to display the characteristics choices
    html.Div([html.Label('Please select the wanted dataset'),
              dcc.Dropdown(id='dataset-dropdown',
                           options=[{'label': name, 'value': method} for name, method in dataset_select.items()],
                           value=list(dataset_select.values())[0])],
             style={'font-size': '20px', 'margin': '10px'}),
    html.Div([html.Label('Please select the dimension reduction method'),
              dcc.RadioItems(id='dimension-reduction-dropdown',
                             options=[{'label': name, 'value': method} for name, method in dim_red_methods.items()],
                             value='pca')]),
    html.Div([html.Label('Please select the clustering method'),
              dcc.RadioItems(id='clustering-method',
                             options=[{'label': name, 'value': method} for name, method in clustering_method_select.items()],
                             value='kmeans')]),
    html.Div([html.Label('Please select the order of operations'),
              dcc.RadioItems(id='order-dropdown',
                             options=[{'label': name, 'value': method} for name, method in order_select.items()],
                             value='dr_clustering')]),
    html.Div(id='kmeans-options', children=[html.Label("Number of Clusters for KMeans:"),
                                            dcc.Input(id='kmeans-clusters', type='number', value=3, style={'margin': '10px'})],
             style={'margin': '10px'}),
    html.Div(id='dbscan-options', children=[html.Label("DBSCAN Parameters:"),
                                            html.Div([html.Label("Epsilon:"),
                                                      dcc.Input(id='dbscan-eps', type='number', value=0.5, style={'margin': '10px'})]),
                                            html.Div([html.Label("Min Samples:"),
                                                      dcc.Input(id='dbscan-min-samples', type='number', value=5, style={'margin': '10px'})])],
             style={'display': 'none', 'margin': '10px'}),
    html.Div(id='hierarchical-options', children=[html.Label("Number of Clusters for Hierarchical Clustering:"),
                                                  dcc.Input(id='hierarchical-clusters', type='number', value=3, style={'margin': '10px'})],
             style={'display': 'none', 'margin': '10px'}),
    html.Button("Recalculate/Load Results", id='recalculate', n_clicks=0, style={'margin': '10px'}),
    dcc.Loading(id="loading-spinner", children=[html.Div(id='output-message'),
                                                html.Div([dcc.Graph(id='tsne-clustering-plot', style={'height': '700px'})],
                                                         style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center'})])
])

# Callback to populate the characteristics dropdown based on the selected dataset
@app.callback(
    Output('characteristics-dropdown', 'options'),
    Input('dataset-dropdown', 'value')
)
def update_characteristics_options(selected_dataset):
    """
    Populate the characteristics dropdown based on the selected dataset.
    """
    # Choose the appropriate dataset
    try:
        if selected_dataset == 'Small Dataset':
            df = pd.read_csv('celeba/celeba_buffalo_s.csv')
        elif selected_dataset == 'Large Dataset':
            df = pd.read_csv('celeba/celeba_buffalo_l.csv')
        else:
            raise PreventUpdate  # No valid dataset selected

        # Extract characteristic columns (assumes characteristics start from the second column)
        characteristic_columns = df.columns[1:]  # Adjust the index if characteristics don't start from the second column
        return [{'label': col, 'value': col} for col in characteristic_columns]
    except Exception as e:
        return []  # Return empty options on failure

# Callback to toggle clustering options based on selected method
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

# Main callback to handle clustering and dimension reduction
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
     State('hierarchical-clusters', 'value'),
     State('order-dropdown', 'value'),
     State('characteristics-dropdown', 'value')]  # Add the characteristics-dropdown input
)
def process_clustering(n_clicks, dataset, dim_red, clustering_method, k_clusters, dbscan_eps, dbscan_min_samples, hierarchical_clusters, order, selected_features):
    if n_clicks == 0:
        return dash.no_update, dash.no_update

    try:
        df = pd.read_csv(dataset)

        # Filter the dataset based on selected features
        if selected_features:
            features = df[selected_features]
        else:
            features = df.iloc[:, 1:40]  # Default feature columns

        embeddings = df.iloc[:, 40:-1]  # Assuming embeddings are the last columns (adjust as needed)
        data = pd.concat([features, embeddings], axis=1)

        if order == 'dr_clustering':
            if dim_red == 'pca':
                reducer = PCA(n_components=2)
            elif dim_red == 'tsne':
                reducer = TSNE(n_components=2)
            else:
                return "Invalid dimension reduction method.", dash.no_update

            reduced_data = reducer.fit_transform(data)
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
            if dim_red == 'pca':
                reducer = PCA(n_components=2)
            elif dim_red == 'tsne':
                reducer = TSNE(n_components=2)
            else:
                return "Invalid dimension reduction method.", dash.no_update

            reduced_data = reducer.fit_transform(features)
            df['Dim 1'], df['Dim 2'] = reduced_data[:, 0], reduced_data[:, 1]

        fig = px.scatter(
            df,
            x='Dim 1',
            y='Dim 2',
            color=df['Cluster'].astype(str),
            title="Clustering Results"
        )

        # Update the layout to enforce a square plot
        fig.update_layout(
            width=1200,  # Set a square width
            height=1200,  # Set a square height
            xaxis=dict(scaleanchor="y",  # Link x-axis scale to y-axis
                       title="Dim 1"),
            yaxis=dict(title="Dim 2")
        )

        return "Clustering completed successfully.", fig
    except Exception as e:
        return f"Error: {e}", dash.no_update

if __name__ == '__main__':
    app.run_server(debug=True)
