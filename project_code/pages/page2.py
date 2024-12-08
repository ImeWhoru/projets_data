# pages/page2.py

from dash import html
from functions import get_theme_styles

def render_page_2(theme: str, dataset: str, data) -> html.Div:
    """
    Rendu de la page 2.
    Args:
        theme (str): Le thème choisi ('light' ou 'dark').
        dataset (str): Le dataset sélectionné ('small' ou 'large').
        data: Les données à afficher.
    Returns:
        html.Div: La structure HTML de la page 2.
    """
    return html.Div([
        html.H1("Page 2", style={'textAlign': 'center'}),
        html.P(f"Theme: {theme.capitalize()}, Dataset: {dataset.capitalize()}."), 
        html.P(f"Dataset contains {len(data)} rows."),
        html.P("This is the content of Page 2."),
        html.Div(style=get_theme_styles(theme))
    ])
