import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dash import Dash, html, dcc # visit http://127.0.0.1:8050/ in your web browser. And ctrl+c to stop the server.
import plotly.express as px

###################################################
##### DEFINITIONS OF FUNCTIONS                #####
###################################################
def generate_table(dataframe, max_rows=10, colorheads='#a27dad', colordata='#dec3e6'):
    return html.Table([
        # Header of the table
        html.Thead(
            html.Tr([html.Th(col, style={'color': colorheads, 'border': '5px solid gray', 'padding': '5px'}) 
                     for col in dataframe.columns])
        ),
        # Body of the table
        html.Tbody([
            # Rows of the table
            html.Tr([
                html.Td(dataframe.iloc[i][col], style={'color': colordata, 'border': '3px solid gray', 'padding': '5px'}) 
                for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ], style={'border-collapse': 'collapse', 'width': '100%'})

markdown_text = '''
### Dash and Markdown

Dash apps can be written in Markdown.
Dash uses the [CommonMark](http://commonmark.org/)
specification of Markdown.
Check out their [60 Second Markdown Tutorial](http://commonmark.org/help/)
if this is your first introduction to Markdown!
'''

###################################################
##### 1 ST STEP: EXTRACT AND ANALYSE THE DATA #####
###################################################

# Load the data from the uploaded CSV file
file_path = 'abalone.csv'
df = pd.read_csv(file_path)

dfcolumns = df.columns.tolist()

# Display the rows of the dataframe
# print(df.head(10)) # Display the first 10 rows
# print(df.tail(10)) # Display the last 10 rows

# Display the shape of the dataframe
# print(f"The dataframe has {df.shape[0]} rows and {df.shape[1]} columns.")

# Display the column names of the dataframe
# print(f"The column names are: {df.columns.tolist()}")

# Display the data types of the columns in the dataframe
# print(df.dtypes)

# Display the summary statistics of the dataframe
# print(df.describe(include='all'))

# Display the number of missing values in each column of the dataframe
# if df.isnull().sum().sum() == 0:
#     print("There are no missing values in the dataframe.")
# else:
#     print("There are missing values in the dataframe.")
#     print(df.isnull().sum())

###################################################
##### 2 ND STEP: BUIL A BASIC LAYOUT          #####
###################################################
# Layout describes what the application looks like

app = Dash(__name__)

colors = {
    'background': '#111111',
    'text': '#7FDBFF',
    'orange': '#FFA500',
    'turquoise': '#40E0D0',
    'pastel turquoise': '#AFEEEE',
}

fig = px.scatter(df, x="Length", y="Diameter", color="Rings")

fig.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)

app.layout = html.Div(style={'backgroundColor': colors['background'],'display': 'flex', 'flexDirection': 'row'}, 
                      children=[
                                html.Div(children=[ # First column of the layout ( subplt )
                                    # Title of the page
                                    html.H1(children='Title of the page', style={
                                                                                    'textAlign': 'center', 
                                                                                    'color': colors['text']
                                                                                    }),
                                    # Subtitle of the section
                                    html.H2(children='Hello Dash', style={
                                                                            'textAlign': 'left', 
                                                                            'color': colors['text']
                                                                            }),
                                    # Description of the application
                                    html.Div(children='''
                                        Dash: A web application framework for your data.
                                    ''', style={
                                                'color': colors['orange']
                                                }),

                                    # Dash Core Components (DCC)
                                    # 1. Graph
                                    dcc.Graph(
                                                id='example-graph',
                                                figure=fig
                                                ),
                                    # Subtitle of the section
                                    html.H2(children='Data Table', style={
                                                                            'textAlign': 'left', 
                                                                            'color': colors['text']
                                                                            }),
                                    # Table component
                                    generate_table(df, colorheads=colors['turquoise'], colordata=colors['pastel turquoise']),
                                ], style={'padding': 100, 'flex': 5}),

                                html.Div(children=[ # Second column of the layout ( subplt )
                                    html.Br(),
                                    html.Br(),
                                    html.Br(), # Replace this by centering vertically the content and tracking the width of the screen and when I resize the screen.
                                    html.Br(), # Track when I go down the page
                                    html.Br(),
                                    html.Br(),

                                    # 2. Markdown
                                    dcc.Markdown(children=markdown_text, style={'color': colors['text']}),

                                    # Make a space
                                    html.Br(),

                                    # 3. Dropdown
                                    html.Label('Dropdown', style={'color': colors['text']}),
                                    dcc.Dropdown(dfcolumns, dfcolumns[0], style={'color': colors['text']}),
                                    html.Br(),

                                    # 4. Multi Dropdown
                                    html.Label('Multi Dropdown', style={'color': colors['text']}),
                                    dcc.Dropdown(dfcolumns, dfcolumns[:2], multi=True, style={'color': colors['text']}),
                                    html.Br(),

                                    # 5. Radio Items
                                    html.Label('Radio Items', style={'color': colors['text']}),
                                    dcc.RadioItems(dfcolumns, dfcolumns[0], style={'color': colors['text']}),
                                    html.Br(),

                                    # 6. Checklist
                                    html.Label('Checklist', style={'color': colors['text']}),
                                    dcc.Checklist(dfcolumns, dfcolumns[:2], style={'color': colors['text']}),
                                    html.Br(),

                                    # 7. Input Text
                                    html.Label('Input Text\t', style={'color': colors['text']}),
                                    dcc.Input(type='text', style={'color': colors['text']}),
                                    html.Br(),

                                    # 8. Slider
                                    html.Label('Slider', style={'color': colors['text']}),
                                    dcc.Slider(min=0, max=5, step=0.5), # Add a color parameter? Change the color as we increase the value?
                                    html.Br()
                                ], style={'padding': 40, 'flex': 3})

                            ])

if __name__ == '__main__':
    app.run(debug=True)