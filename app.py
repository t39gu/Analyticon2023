import os
import json
import requests
import plotly.express as px
import numpy as np
import pandas as pd

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

datasets_path = './datasets/pokedex_(Update_05.20).csv'
logo_url = 'https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/25.png'
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

pokemon_df = pd.read_csv(datasets_path)
pokemon_options = [{"label": name, "value": name} for name in pokemon_df.name.unique()]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SKETCHY, external_stylesheets])

application = app.server
app.config['suppress_callback_exceptions'] = True

logo = html.Img(src=logo_url, height='100vh')

navbar = dbc.NavbarSimple(
    children=([
        dbc.Row([
            dbc.Col([
                dbc.NavItem(logo)
            ]),
            dbc.Col([
                dbc.DropdownMenu([
                    dbc.DropdownMenuItem(children=["Contact Us"], header=True),
                    dbc.DropdownMenuItem("Jie Chen",
                                         href='https://phonetool.amazon.com/users/jiematth',
                                         target='_blank'),
                    dbc.DropdownMenuItem("Ting Gu",
                                         href='https://phonetool.amazon.com/users/goldmjr',
                                         target='_blank')
                ], nav=True, in_navbar=True, label="Contact Us", direction="start", align_end=True
                )
            ], width={'size': 2, 'offset': 6, 'order': 'last'})
        ], style={'textAlign': 'right'}),
    ]),
    brand='Pokémon Guide',
    color='info',
    fluid=True,
    dark=False,
    brand_style={"fontSize": 36}
)

#Todo: move style into static/styles.css
tab_style = {'borderTop': '1px solid #d6d6d6', 'padding': '6px', 'fontWeight': 'bold'}
image_style = {'width': '300px', 'height': '300px'}
app_style = {'marginLeft': 40, 'marginRight': 40, 'marginTop': 20, 'marginBottom': 20, 'padding': '10px'}

tab1 = dbc.Tab(label='Pokémon 🥳', tab_id='tab1',
               style=tab_style, activeTabClassName="fw-bold fst-italic",
               children=[
                   html.Br(),
                   html.P('Select a Pokémon'),
                   html.Div(
                       dcc.Dropdown(
                           id='select-pokemon', options=pokemon_options,
                           multi=False,
                           value='Bulbasaur',
                       ), style={'width': '30%'}
                   ),
                   dbc.Row([
                       dbc.Col(
                           id='pokemon-img-front'
                           # html.Img(id='pokemon-img-front',style=image_style)
                           , width=2),
                       dbc.Col(
                           id='pokemon-img-shiny'
                           # html.Img(id='pokemon-img-shiny',style=image_style)
                           , width=3),
                       dbc.Col([
                           html.Div(id='pokemon-desc')
                       ]),
                       dbc.Col([
                           dcc.Graph(id='attributes-plot')
                       ], width=3)

                   ]),
                   dbc.Row([
                       html.P('Evolution Tree')
                   ])
               ])

tab2 = dbc.Tab(label='Create A Pokémon 🎭', tab_id='tab2',
               style=tab_style, activeTabClassName="fw-bold fst-italic",
               children=[
                   dbc.Row([
                       dbc.Col([
                           dbc.Row([
                               html.P('Pokémon #1'),
                               dcc.Dropdown(
                                   id='select-pokemon-1', options=pokemon_options,
                                   multi=False,
                                   value='Bulbasaur'),
                               html.P('Pokémon #2'),
                               dcc.Dropdown(
                                   id='select-pokemon-2', options=pokemon_options,
                                   multi=False,
                                   value='Bulbasaur')
                           ])
                       ], width=6),
                       dbc.Col([
                           html.P('Your New Pokémon!')
                       ], width=6)

                   ])
               ])

tab3 = dbc.Tab(label='Ask Me A Question 🔮', tab_id='tab3',
               style=tab_style, activeTabClassName="fw-bold fst-italic",
               children=[
                   html.P(),
                   dbc.Input(id='search-input', placeholder="Ask me something about Pokémon", size="lg",
                             className="mb-3"),
                   dbc.Spinner(html.Div(id='search-result')),
               ])

app.layout = html.Div(
    children=[
        navbar,
        html.Br(),
        dbc.Tabs([
            tab1,
            tab2,
            tab3
        ])
    ], style=app_style
)

@app.callback(
    Output('pokemon-img-front', 'children'),
    Output('pokemon-img-shiny', 'children'),
    Output('attributes-plot', 'figure'),
    Output('pokemon-desc', 'children'),
    Input('select-pokemon', 'value')
)
def update_pokemon_info(pokemon_name):
    print(pokemon_name)
    if pokemon_name is None:
        PreventUpdate
    else:
        sub_df = pokemon_df[pokemon_df.name == pokemon_name]
        pokedex = sub_df.pokedex_number.unique()[0]
        sub_group = pokemon_df[pokemon_df.pokedex_number == pokedex]
        # print(sub_group.name.unique())
        # if sub_group.shape[0]>1:
        #     pokemon_fig_name = sub_group.name.unique()[0]
        # else:
        #     pokemon_fig_name = pokemon_name
        # if pokemon_fig_name.endswith('Female'): 
        #     pokemon_fig_name = pokemon_fig_name.replace(' Female','-female')
        # elif pokemon_fig_name.endswith('Male'):
        #     pokemon_fig_name = pokemon_fig_name.replace(' Male','-male')
        # elif pokemon_fig_name.endswith('♀'): 
        #     pokemon_fig_name = pokemon_fig_name.replace('♀','-f')
        # elif pokemon_fig_name.endswith('♂'): 
        #     pokemon_fig_name = pokemon_fig_name.replace('♂','-m')
        # else:
        #     pokemon_fig_name = pokemon_fig_name
        # print(pokemon_name,pokemon_fig_name)
        url = f'https://pokeapi.co/api/v2/pokemon/{pokedex}'
        response = requests.get(url)
        if response.status_code == 200:
            pokemon_fig_data = response.json()
            image_url_front = pokemon_fig_data['sprites']['other']['official-artwork']['front_default']
            image_url_shiny = pokemon_fig_data['sprites']['other']['official-artwork']['front_shiny']
            front_content = html.Img(src=image_url_front, style=image_style)
            shiny_content = html.Img(src=image_url_shiny, style=image_style)
        else:
            front_content = html.P('Pokemon Image Not Found', style={'color': '#957DAD', 'fontSize': 14})
            shiny_content = html.P('Shiny Image Not Found', style={'color': '#D291BC', 'fontSize': 14})
        plot_df = sub_df.loc[:, ['hp', 'attack', 'defense', 'speed', 'sp_attack', 'sp_defense']]
        plot_df.columns = ['HP', 'Attack', 'Defense', 'Speed', 'Sp. Attack', 'Sp. Defense']
        plot_df2 = plot_df.transpose().reset_index(drop=False)
        plot_df2.columns = ['type', 'points']
        max_rg = plot_df2.points.max() + 20
        attribute_fig = px.line_polar(
            plot_df2,
            r='points',
            theta='type', line_close=True,
            range_r=(0, max_rg),
            title=f'{pokemon_name} Attributes',
            template='seaborn',
            color_discrete_sequence=['dodgerblue'],
        )
        attribute_fig.update_traces(fill='toself')
        attribute_fig.update_layout(margin={"r": 30, "t": 0, "l": 50, "b": 0, 'pad': 4})
        attribute_fig.update_layout(font=dict(size=8, color='RebeccaPurple'))
        num_ref = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight'}
        gen_num = sub_df.generation.unique()[0]
        gen = num_ref[gen_num]
        if sub_df.type_number.unique()[0] == 1:
            type_bar = dbc.Progress([
                dbc.Progress(value=100, label=f'{sub_df.type_1.unique()[0]}', color='#9699CB', bar=True)],
                style={"height": "50px"}),
        else:
            type_bar = dbc.Progress([
                dbc.Progress(value=100, label=f'{sub_df.type_1.unique()[0]}', color='#9699CB', bar=True),
                dbc.Progress(value=100, label=f'{sub_df.type_2.unique()[0]}', color='#D18CAF', bar=True)
            ], style={"height": "50px"}),
        if sub_df.abilities_number.unique()[0] == 0:
            ability_bar = dbc.Progress([
                dbc.Progress(value=100, label='No ability', color='#A9D4B7', bar=True)], style={"height": "50px"}),
        elif sub_df.abilities_number.unique()[0] == 1:
            ability_bar = dbc.Progress([
                dbc.Progress(value=100, label=f'{sub_df.ability_1.unique()[0]}', color='#A9D4B7', bar=True)],
                style={"height": "50px"}),
        elif sub_df.abilities_number.unique()[0] == 2:
            ability_bar = dbc.Progress([
                dbc.Progress(value=100, label=f'{sub_df.ability_1.unique()[0]}', color='#A9D4B7', bar=True),
                dbc.Progress(value=100, label=f'{sub_df.ability_hidden.unique()[0]}', color='#868D8B', bar=True),
            ], style={"height": "50px"})
        else:
            ability_bar = dbc.Progress([
                dbc.Progress(value=100, label=f'{sub_df.ability_1.unique()[0]}', color='#A9D4B7', bar=True),
                dbc.Progress(value=100, label=f'{sub_df.ability_2.unique()[0]}', color='#868D8B', bar=True),
                dbc.Progress(value=100, label=f'{sub_df.ability_hidden.unique()[0]}', color='#A7C0CF', bar=True)
            ], style={"height": "50px"})
        return front_content, shiny_content, attribute_fig, \
            [
                html.P(f'Generation: {gen}'),
                html.P(f'Status: {sub_df.status.unique()[0]}'),
                html.P(f'Species: {sub_df.species.unique()[0]}'),
                html.Div('Type: '),
                html.Div(children=type_bar),
                html.P(),
                html.Div('Abilities: '),
                html.Div(children=ability_bar)
            ]


if __name__ == '__main__':
    application.run(debug=False, port=8050)
