# functions.py
from dash import html, dcc

THEMES = {
    'light': {
        'background-color': '#f4f4f4',
        'sidebar-color': '#f9f9f9',
        'color': '#000',
        'text-color': '#000',
        'title-color': '#000',
        'box-color': '#fff',
        'titlebg-color': '#e6e6e6',

        'boxes-color': '#DEDFE4',
        'boxes-hover-color': '#D0D1D5',
        'boxes-selected-color': '#C2C3C7',
        
        
    },
    'dark': {
        'background-color': '#333',
        'sidebar-color': '#222',
        'color': '#fff',
        'text-color': '#fff',
        'title-color': '#fff',
        'box-color': '#444',
        'titlebg-color': '#111',

        'boxes-color': '#DEDFE4',
        'boxes-hover-color': '#D0D1D5',
        'boxes-selected-color': '#C2C3C7',
    }
}

def get_theme_styles(theme: str) -> dict:
    return THEMES.get(theme, THEMES['light'])

def show_sidebar(sidebar_state: str, theme: str, dataset: str, current_page: str = "1") -> tuple:
    theme_styles = get_theme_styles(theme)

    sidebar_style = {
        'position': 'fixed',
        'height': '100%',
        'background-color': theme_styles['sidebar-color'],
        'color': theme_styles['color'],
        'box-shadow': '2px 0px 5px rgba(0,0,0,0.1)'
    }
    content_style = {
        'background-color': theme_styles['background-color'],
        'color': theme_styles['color'],
        'height': '1500px',
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
            html.Hr(style={'border-top': '2px solid'}),
            html.H4(f"Current Page: {current_page}", style={'color': theme_styles['color']}),
            html.Hr(style={'border-top': '2px solid'}),
            html.A("Page 1", href="/page-1", style={'display': 'block', 'margin-top': '10px'}),
            html.A("Page 2", href="/page-2", style={'display': 'block', 'margin-top': '10px'}),
        ])

    return sidebar_style, content_style, sidebar_state, sidebar_content
