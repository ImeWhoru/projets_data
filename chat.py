from dash import Dash, dcc, html, Input, Output

# Initialize the app
app = Dash(__name__)

# Define layouts for different pages
page_1_layout = html.Div([
    html.H1("Page 1"),
    dcc.Link("Go to Page 2", href="/page-2"),
    html.Br(),
    dcc.Link("Go to Home", href="/")
])

page_2_layout = html.Div([
    html.H1("Page 2"),
    dcc.Link("Go to Page 1", href="/page-1"),
    html.Br(),
    dcc.Link("Go to Home", href="/")
])

home_layout = html.Div([
    html.H1("Home Page"),
    dcc.Link("Go to Page 1", href="/page-1"),
    html.Br(),
    dcc.Link("Go to Page 2", href="/page-2")
])

# Define the app layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),  # Tracks the URL
    html.Div(id='page-content')  # Content updated dynamically
])

# Callback to update the page content based on URL
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/page-1':
        return page_1_layout
    elif pathname == '/page-2':
        return page_2_layout
    else:
        return home_layout

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
