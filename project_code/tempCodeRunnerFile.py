    # Slider-style dataset selector
    html.Div([
        html.Label("Dataset Selector", style={
            'textAlign': 'center', 
            'font-weight': 'bold', 
            'margin-bottom': '10px'
        }),
        dcc.Slider(
            id='dataset-slider',
            min=0,
            max=1,
            marks={
                0: {'label': 'Small', 'style': {'color': 'green', 'font-weight': 'bold'}},
                1: {'label': 'Large', 'style': {'color': 'red', 'font-weight': 'bold'}}
            },
            value=0,  # Default to 'Small'
            tooltip={"placement": "bottom", "always_visible": True},
            updatemode='drag'  # Updates as you drag
        )
    ], style={
        'width': '50%', 
        'margin': '20px auto', 
        'padding': '20px', 
        'border': '2px solid black', 
        'border-radius': '15px', 
        'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.2)', 
        'background-color': '#f9f9f9', 
        'textAlign': 'center'
    })
