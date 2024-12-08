# functions.py

from dash import html, dcc

# Centralisation des thèmes
THEMES = {
    'light': {
        'background-color': '#f4f4f4',
        'sidebar-color': '#f9f9f9',
        'color': '#000',
        'text-color': '#000'
    },
    'dark': {
        'background-color': '#333',
        'sidebar-color': '#222',
        'color': '#fff',
        'text-color': '#fff'
    }
}

def get_theme_styles(theme: str) -> dict:
    """
    Retourne les styles associés à un thème donné.
    """
    return THEMES.get(theme, THEMES['light'])  # Par défaut : thème clair

# def get_sidebar_content(theme: str, dataset: str):
#     """
#     Returns the content of the sidebar based on the current theme and dataset.
#     """
#     return html.Div([
#         html.H2("Settings", style={'textAlign': 'center', 'margin-bottom': '20px'}),
#         html.Label("Theme Selector", style={'font-weight': 'bold'}),
#         dcc.RadioItems(
#             id='theme-toggle',
#             options=[
#                 {'label': 'Light Mode', 'value': 'light'},
#                 {'label': 'Dark Mode', 'value': 'dark'}
#             ],
#             value=theme,
#             style={'margin-bottom': '20px'}
#         ),
#         html.Hr(),
#         html.Label("Dataset Selector", style={'font-weight': 'bold'}),
#         dcc.RadioItems(
#             id='dataset-selector',
#             options=[
#                 {'label': 'Small Dataset', 'value': 'small'},
#                 {'label': 'Large Dataset', 'value': 'large'}
#             ],
#             value=dataset,
#             style={'margin-bottom': '20px'}
#         ),
#         html.A("Page 1", href="/page-1", style={'display': 'block', 'margin-top': '10px'}),
#         html.A("Page 2", href="/page-2", style={'display': 'block', 'margin-top': '10px'}),
#     ])

def show_sidebar(sidebar_state: str, theme: str, dataset: str) -> tuple:
    """
    Updates the sidebar and content styles based on the sidebar state and theme.
    """
    theme_styles = get_theme_styles(theme)

    sidebar_style = {
        'position': 'fixed',
        'height': '100%',
        'background-color': theme_styles['sidebar-color'],
        'color': theme_styles['color'],
        'box-shadow': '2px 0px 5px rgba(0,0,0,0.1)'
    }
    content_style = {
        'padding': '20px',
        'background-color': theme_styles['background-color'],
        'color': theme_styles['color'],
        'height': '100%',
    }

    if sidebar_state == 'collapsed':
        sidebar_style.update({'width': '3%', 'padding': '10px'})
        content_style.update({'margin-left': '5%'})
        sidebar_content = None
    else:
        sidebar_style.update({'width': '10%', 'padding': '20px'})
        content_style.update({'margin-left': '12%'})
        sidebar_content = html.Div([
            html.H2("Settings", style={'textAlign': 'center', 'margin-bottom': '20px'}),
            html.Label("Theme Selector", style={'font-weight': 'bold'}),
            dcc.RadioItems(
                id='theme-toggle',
                options=[
                    {'label': 'Light Mode', 'value': 'light'},
                    {'label': 'Dark Mode', 'value': 'dark'}
                ],
                value=theme,
                style={'margin-bottom': '20px'}
            ),
            html.Hr(),
            html.Label("Dataset Selector", style={'font-weight': 'bold'}),
            dcc.RadioItems(
                id='dataset-selector',
                options=[
                    {'label': 'Small Dataset', 'value': 'small'},
                    {'label': 'Large Dataset', 'value': 'large'}
                ],
                value=dataset,
                style={'margin-bottom': '20px'}
            ),
            html.A("Page 1", href="/page-1", style={'display': 'block', 'margin-top': '10px'}),
            html.A("Page 2", href="/page-2", style={'display': 'block', 'margin-top': '10px'}),
        ])

    return sidebar_style, content_style, sidebar_state, sidebar_content
