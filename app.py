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
import dash_loading_spinners as dls

datasets_path = './datasets/pokedex_(Update_05.20).csv'
logo_url = 'https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/25.png'
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

pokemon_df = pd.read_csv(datasets_path)
pokemon_options = [{"label": name, "value": name} for name in pokemon_df.name.unique()]

def get_evolution_chain(pokemon_id):
    
    # Get the species to find the evolution chain URL
    species_url = f"https://pokeapi.co/api/v2/pokemon-species/{pokemon_id}/"
    response = requests.get(species_url)
    data = response.json()
    evolution_url = data['evolution_chain']['url']
    
    # Get the evolution chain data
    response = requests.get(evolution_url)
    data = response.json()
    chain = data['chain']

    # Extract evolution names from the chain
    evolutions = [chain['species']['name']]
    while chain['evolves_to']:
        evolutions.append(chain['evolves_to'][0]['species']['name'])
        chain = chain['evolves_to'][0]
    
    return evolutions

def get_difference_emoji(value):
    if value > 0:
        return "🔺"
    elif value < 0:
        return "🔻"
    else:
        return "➡️"

plot_df = pokemon_df.groupby(['generation','status'])['name'].count().reset_index(drop=False)
plot_df = plot_df.rename(columns={'name':'count'})
plot_df['percent'] = pokemon_df.groupby(['generation','status']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
plot_df['percent'] = plot_df['percent'].round(decimals=2)
generation_plot1 = px.bar(plot_df, x="generation", y='percent',text='percent',
             color='status',color_discrete_sequence=px.colors.qualitative.T10,
             category_orders={"status": ["Normal", "Mythical", "Sub Legendary", "Legendary"]})
generation_plot1.update_traces(texttemplate='%{text:.2s}%', textposition='outside')
generation_plot1.update_layout(title_text='Pokémon over Generation',
                  yaxis=dict(title=''),
                  xaxis=dict(
                    title='Generation',
                    titlefont_size=16,
                    tickfont_size=14,
                ),
                uniformtext_minsize=8,
                yaxis_ticksuffix = "%",
                hovermode="x unified"
    )

generation_plot2 = px.box(pokemon_df, x="generation", y='total_points',points='all',
             color='status',#notched=True,
             color_discrete_sequence=px.colors.qualitative.T10,
             custom_data=[pokemon_df['name'],pokemon_df['status']],
             )
generation_plot2.update_layout(title_text='Pokémon Points over Generation',
                  yaxis=dict(title=''),
                  xaxis=dict(
                    title='Generation',
                    titlefont_size=16,
                    tickfont_size=14,
                ),
                uniformtext_minsize=8,
    )
generation_plot2.update_traces(hovertemplate='Status:%{customdata[1]}<br>Generation:%{x}<br>Total Points:%{y}<br>Name:%{customdata[0]}')


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
                   html.Div([
                       html.P('Select a Pokémon ',style=dict(width='20%',fontSize=24)),
                        html.Div(
                            dcc.Dropdown(
                                id='select-pokemon', options=pokemon_options,
                                multi=False,
                                value='Bulbasaur',
                            ), style={'width': '25%','font-size':24}
                        ),
                   ],style=dict(display='flex')),
                   dbc.Button('General Stats',color='secondary',size='lg',outline=False,
                              style={'font-size':'30px','width':'100%'}),
                   dbc.Row([
                       html.Div(html.Br()),
                       dbc.Col(
                           dls.Hash(html.Div(
                           id='pokemon-img-front'
                           ),color='#435278',speed_multiplier=2,size=30,fullscreen=False)
                           ),
                       dbc.Col(
                           dls.Hash(html.Div(
                           id='pokemon-img-shiny'
                           ),color='#435278',speed_multiplier=2,size=30,fullscreen=False)
                           ),
                       dbc.Col([
                           dls.Hash(
                           html.Div(id='pokemon-desc'),color='#435278',speed_multiplier=2,size=30,fullscreen=False)
                       ]),
                       dbc.Col([
                           dls.Hash(
                           dcc.Graph(id='attributes-plot'),color='#435278',speed_multiplier=2,size=30,fullscreen=False)
                       ])

                   ]),
                   html.Div(html.Br()),
                   dbc.Button('Evolution Tree',color='secondary',size='lg',outline=False,
                              style={'font-size':'30px','width':'100%'}),
                   html.Div(html.Br()),
                   dbc.Row([
                       dls.Hash(
                       html.Div(id='evolution-tree'),color='#435278',speed_multiplier=2,size=30,fullscreen=False)
                   ]),
                   html.Div(html.Br()),
                   dbc.Button('Comparison Across Generation',color='secondary',size='lg',outline=False,
                              style={'font-size':'30px','width':'100%'}),
                   html.Div(html.Br()),
                   dbc.Row([
                       dbc.Col([
                          dls.Hash(
                           dcc.Graph(id='generation-plot1',figure=generation_plot1),color='#435278',speed_multiplier=2,size=30,fullscreen=False)
                       ]),
                       dbc.Col([
                          dls.Hash(
                           dcc.Graph(id='generation-plot2',figure=generation_plot2),color='#435278',speed_multiplier=2,size=30,fullscreen=False)
                           
                       ])
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

# tab3 = dbc.Tab(label='Ask Me A Question 🔮', tab_id='tab3',
#                style=tab_style, activeTabClassName="fw-bold fst-italic",
#                children=[
#                    html.P(),
#                    dbc.Input(id='search-input', placeholder="Ask me something about Pokémon", size="lg",
#                              className="mb-3"),
#                    dbc.Spinner(html.Div(id='search-result')),
#                ])

app.layout = html.Div(
    children=[
        navbar,
        html.Br(),
        dbc.Tabs([
            tab1,
            tab2,
            #tab3
        ])
    ], style=app_style
)

@app.callback(
    Output('pokemon-img-front', 'children'),
    Output('pokemon-img-shiny', 'children'),
    Output('attributes-plot', 'figure'),
    Output('pokemon-desc', 'children'),
    Output('evolution-tree','children'),
    Input('select-pokemon', 'value')
)
def update_pokemon_info(pokemon_name):
    if pokemon_name is None:
        PreventUpdate
    else:
        sub_df = pokemon_df[pokemon_df.name == pokemon_name]
        pokedex = sub_df.pokedex_number.unique()[0]
        url = f'https://pokeapi.co/api/v2/pokemon/{pokedex}'
        response = requests.get(url)
        if response.status_code == 200:
            pokemon_fig_data = response.json()
            image_url_front = pokemon_fig_data['sprites']['other']['official-artwork']['front_default']
            image_url_shiny = pokemon_fig_data['sprites']['other']['official-artwork']['front_shiny']
            front_content = html.Div([
                html.H5('Regular',style={"text-align": "center"}),
                html.Img(src=image_url_front, style=image_style)
            ])
            shiny_content = html.Div([
                html.H5('Shiny',style={"text-align": "center"}),
                html.Img(src=image_url_shiny, style=image_style)
            ])
        else:
            front_content = html.P('Pokemon Image Not Found', style={'color': '#957DAD', 'fontSize': 24})
            shiny_content = html.P('Shiny Image Not Found', style={'color': '#D291BC', 'fontSize': 24})
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
        gender_bar = dbc.Progress([
            dbc.Progress(value=sub_df.percentage_male.unique()[0],
                         label=f'♂: {sub_df.percentage_male.unique()[0]}',color='#DF917C',bar=True),
            dbc.Progress(value=100-sub_df.percentage_male.unique()[0],
                         label=f'♀: {100-sub_df.percentage_male.unique()[0]}',color='#C7DBDA',bar=True)
        ], style={"height": "50px"})
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
        evolution_chain = get_evolution_chain(pokedex)
        if len(evolution_chain)<2:
            evol_tree = html.P('No evolution chain found!')
        elif len(evolution_chain)==2:
            
            index1 = pokemon_df[pokemon_df.name.str.lower() == evolution_chain[0]].pokedex_number.unique()[0]
            index2 = pokemon_df[pokemon_df.name.str.lower() == evolution_chain[1]].pokedex_number.unique()[0]
            url1 = f'https://pokeapi.co/api/v2/pokemon/{index1}'
            url2 = f'https://pokeapi.co/api/v2/pokemon/{index2}'
            response1 = requests.get(url1)
            response2 = requests.get(url2)
            if response1.status_code == 200 and response2.status_code==200:
                hp1 = pokemon_df[pokemon_df.name.str.lower() == evolution_chain[0]].hp.unique()[0]
                attack1=pokemon_df[pokemon_df.name.str.lower() == evolution_chain[0]].attack.unique()[0]
                defense1=pokemon_df[pokemon_df.name.str.lower() == evolution_chain[0]].defense.unique()[0]
                speed1 = pokemon_df[pokemon_df.name.str.lower() == evolution_chain[0]].speed.unique()[0]
                spattack1=pokemon_df[pokemon_df.name.str.lower() == evolution_chain[0]].sp_attack.unique()[0]
                spdefense1=pokemon_df[pokemon_df.name.str.lower() == evolution_chain[0]].sp_defense.unique()[0]

                hp2 = pokemon_df[pokemon_df.name.str.lower() == evolution_chain[1]].hp.unique()[0]
                attack2=pokemon_df[pokemon_df.name.str.lower() == evolution_chain[1]].attack.unique()[0]
                defense2=pokemon_df[pokemon_df.name.str.lower() == evolution_chain[1]].defense.unique()[0]
                speed2 = pokemon_df[pokemon_df.name.str.lower() == evolution_chain[1]].speed.unique()[0]
                spattack2=pokemon_df[pokemon_df.name.str.lower() == evolution_chain[1]].sp_attack.unique()[0]
                spdefense2=pokemon_df[pokemon_df.name.str.lower() == evolution_chain[1]].sp_defense.unique()[0]
                
                hp_diff1_2 = hp2-hp1
                attack_diff1_2 = attack2-attack1
                defense_diff1_2 = defense2-defense1
                speed_diff1_2 = speed2-speed1
                spattack_diff1_2 = spattack2-spattack1
                spdefense_diff1_2 = spdefense2-spdefense1

                pokemon_fig_data1 = response1.json()
                image_url_front1 = pokemon_fig_data1['sprites']['other']['official-artwork']['front_default']
                pokemon_fig_data2 = response2.json()
                image_url_front2 = pokemon_fig_data2['sprites']['other']['official-artwork']['front_default']
                evol_tree = html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.H5(f'{evolution_chain[0].capitalize()}',style={"text-align": "center"}),
                            html.Img(src=image_url_front1, style=image_style),
                            dash_table.DataTable(
                                columns=[
                                    {"name": "Attribute", "id": "attribute"},
                                    {"name": evolution_chain[0].capitalize(), "id": "pokemon_1"}
                                ],
                                data=[
                                    {"attribute": "HP", "pokemon_1": hp1},
                                    {"attribute": "Attack", "pokemon_1": attack1},
                                    {"attribute": "Defense", "pokemon_1": defense1},
                                    {"attribute": "Speed", "pokemon_1": speed1},
                                    {"attribute": "Sp. Attack", "pokemon_1": spattack1},
                                    {"attribute": "Sp. Defense", "pokemon_1": spdefense1}
                                ]
                            ),
                        ]),
                        dbc.Col([
                            html.Button(children=[html.Img(src='https://static.thenounproject.com/png/6402-84.png')]),
                        ],width=1,align="center"),
                        dbc.Col([
                            html.H5(f'{evolution_chain[1].capitalize()}',style={"text-align": "center"}),
                            html.Img(src=image_url_front2, style=image_style),
                            dash_table.DataTable(
                                columns=[
                                    {"name": "Attribute", "id": "attribute"},
                                    {"name": evolution_chain[1].capitalize(), "id": "pokemon_1"},
                                    {"name":'Change','id':'change'}
                                ],
                                data=[
                                    {"attribute": "HP", "pokemon_1": hp2,
                                     'change': f"{get_difference_emoji(hp_diff1_2)} {abs(hp_diff1_2):.0f}"
                                    },
                                    {"attribute": "Attack", "pokemon_1": attack2,
                                     'change': f"{get_difference_emoji(attack_diff1_2)} {abs(attack_diff1_2):.0f}"},
                                    {"attribute": "Defense", "pokemon_1": defense2,
                                     'change': f"{get_difference_emoji(defense_diff1_2)} {abs(defense_diff1_2):.0f}"},
                                    {"attribute": "Speed", "pokemon_1": speed2,
                                     'change': f"{get_difference_emoji(speed_diff1_2)} {abs(speed_diff1_2):.0f}"},
                                    {"attribute": "Sp. Attack", "pokemon_1": spattack2,
                                     'change': f"{get_difference_emoji(spattack_diff1_2)} {abs(spattack_diff1_2):.0f}"},
                                    {"attribute": "Sp. Defense", "pokemon_1": spdefense2,
                                     'change': f"{get_difference_emoji(spdefense_diff1_2)} {abs(spdefense_diff1_2):.0f}"},
                                ]
                            ),
                        ])
                    ])
                ])
            else:
                evol_tree = html.P('No evolution chain found!')
        else:
            index1 = pokemon_df[pokemon_df.name.str.lower() == evolution_chain[0]].pokedex_number.unique()[0]
            index2 = pokemon_df[pokemon_df.name.str.lower() == evolution_chain[1]].pokedex_number.unique()[0]
            index3 = pokemon_df[pokemon_df.name.str.lower() == evolution_chain[2]].pokedex_number.unique()[0]
            url1 = f'https://pokeapi.co/api/v2/pokemon/{index1}'
            url2 = f'https://pokeapi.co/api/v2/pokemon/{index2}'
            url3 = f'https://pokeapi.co/api/v2/pokemon/{index3}'
            response1 = requests.get(url1)
            response2 = requests.get(url2)
            response3 = requests.get(url3)
            if response1.status_code == 200 and response2.status_code==200 and response3.status_code==200:
                pokemon_fig_data1 = response1.json()
                image_url_front1 = pokemon_fig_data1['sprites']['other']['official-artwork']['front_default']
                pokemon_fig_data2 = response2.json()
                image_url_front2 = pokemon_fig_data2['sprites']['other']['official-artwork']['front_default']
                pokemon_fig_data3 = response3.json()
                image_url_front3 = pokemon_fig_data3['sprites']['other']['official-artwork']['front_default']

                hp1 = pokemon_df[pokemon_df.name.str.lower() == evolution_chain[0]].hp.unique()[0]
                attack1=pokemon_df[pokemon_df.name.str.lower() == evolution_chain[0]].attack.unique()[0]
                defense1=pokemon_df[pokemon_df.name.str.lower() == evolution_chain[0]].defense.unique()[0]
                speed1 = pokemon_df[pokemon_df.name.str.lower() == evolution_chain[0]].speed.unique()[0]
                spattack1=pokemon_df[pokemon_df.name.str.lower() == evolution_chain[0]].sp_attack.unique()[0]
                spdefense1=pokemon_df[pokemon_df.name.str.lower() == evolution_chain[0]].sp_defense.unique()[0]

                hp2 = pokemon_df[pokemon_df.name.str.lower() == evolution_chain[1]].hp.unique()[0]
                attack2=pokemon_df[pokemon_df.name.str.lower() == evolution_chain[1]].attack.unique()[0]
                defense2=pokemon_df[pokemon_df.name.str.lower() == evolution_chain[1]].defense.unique()[0]
                speed2 = pokemon_df[pokemon_df.name.str.lower() == evolution_chain[1]].speed.unique()[0]
                spattack2=pokemon_df[pokemon_df.name.str.lower() == evolution_chain[1]].sp_attack.unique()[0]
                spdefense2=pokemon_df[pokemon_df.name.str.lower() == evolution_chain[1]].sp_defense.unique()[0]
                
                hp_diff1_2 = hp2-hp1
                attack_diff1_2 = attack2-attack1
                defense_diff1_2 = defense2-defense1
                speed_diff1_2 = speed2-speed1
                spattack_diff1_2 = spattack2-spattack1
                spdefense_diff1_2 = spdefense2-spdefense1

                hp3 = pokemon_df[pokemon_df.name.str.lower() == evolution_chain[2]].hp.unique()[0]
                attack3=pokemon_df[pokemon_df.name.str.lower() == evolution_chain[2]].attack.unique()[0]
                defense3=pokemon_df[pokemon_df.name.str.lower() == evolution_chain[2]].defense.unique()[0]
                speed3 = pokemon_df[pokemon_df.name.str.lower() == evolution_chain[2]].speed.unique()[0]
                spattack3=pokemon_df[pokemon_df.name.str.lower() == evolution_chain[2]].sp_attack.unique()[0]
                spdefense3=pokemon_df[pokemon_df.name.str.lower() == evolution_chain[2]].sp_defense.unique()[0]
                
                hp_diff2_3 = hp3-hp2
                attack_diff2_3 = attack3-attack2
                defense_diff2_3 = defense3-defense2
                speed_diff2_3 = speed3-speed2
                spattack_diff2_3 = spattack3-spattack2
                spdefense_diff2_3 = spdefense3-spdefense2
                evol_tree = html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.H5(f'{evolution_chain[0].capitalize()}',style={"text-align": "center"}),
                            html.Img(src=image_url_front1, style=image_style),
                            dash_table.DataTable(
                                columns=[
                                    {"name": "Attribute", "id": "attribute"},
                                    {"name": evolution_chain[0].capitalize(), "id": "pokemon_1"}
                                ],
                                data=[
                                    {"attribute": "HP", "pokemon_1": hp1},
                                    {"attribute": "Attack", "pokemon_1": attack1},
                                    {"attribute": "Defense", "pokemon_1": defense1},
                                    {"attribute": "Speed", "pokemon_1": speed1},
                                    {"attribute": "Sp. Attack", "pokemon_1": spattack1},
                                    {"attribute": "Sp. Defense", "pokemon_1": spdefense1}
                                ]
                            ),
                        ]),
                        dbc.Col([
                            html.Button(children=[html.Img(src='https://static.thenounproject.com/png/6402-84.png')]),
                        ],width=1,align="center"),
                        dbc.Col([
                            html.H5(f'{evolution_chain[1].capitalize()}',style={"text-align": "center"}),
                            html.Img(src=image_url_front2, style=image_style),
                            dash_table.DataTable(
                                columns=[
                                    {"name": "Attribute", "id": "attribute"},
                                    {"name": evolution_chain[1].capitalize(), "id": "pokemon_1"},
                                    {"name":'Change','id':'change'}
                                ],
                                data=[
                                    {"attribute": "HP", "pokemon_1": hp2,
                                     'change': f"{get_difference_emoji(hp_diff1_2)} {abs(hp_diff1_2):.0f}"
                                    },
                                    {"attribute": "Attack", "pokemon_1": attack2,
                                     'change': f"{get_difference_emoji(attack_diff1_2)} {abs(attack_diff1_2):.0f}"},
                                    {"attribute": "Defense", "pokemon_1": defense2,
                                     'change': f"{get_difference_emoji(defense_diff1_2)} {abs(defense_diff1_2):.0f}"},
                                    {"attribute": "Speed", "pokemon_1": speed2,
                                     'change': f"{get_difference_emoji(speed_diff1_2)} {abs(speed_diff1_2):.0f}"},
                                    {"attribute": "Sp. Attack", "pokemon_1": spattack2,
                                     'change': f"{get_difference_emoji(spattack_diff1_2)} {abs(spattack_diff1_2):.0f}"},
                                    {"attribute": "Sp. Defense", "pokemon_1": spdefense2,
                                     'change': f"{get_difference_emoji(spdefense_diff1_2)} {abs(spdefense_diff1_2):.0f}"},
                                ]
                            ),
                        ]),
                        dbc.Col([
                            html.Button(children=[html.Img(src='https://static.thenounproject.com/png/6402-84.png')]),
                        ],width=1,align="center"),
                        dbc.Col([
                            html.H5(f'{evolution_chain[2].capitalize()}',style={"text-align": "center"}),
                            html.Img(src=image_url_front3, style=image_style),
                            dash_table.DataTable(
                                columns=[
                                    {"name": "Attribute", "id": "attribute"},
                                    {"name": evolution_chain[2].capitalize(), "id": "pokemon_1"},
                                    {"name":'Change','id':'change'}
                                ],
                                data=[
                                    {"attribute": "HP", "pokemon_1": hp3,
                                     'change': f"{get_difference_emoji(hp_diff2_3)} {abs(hp_diff2_3):.0f}"
                                    },
                                    {"attribute": "Attack", "pokemon_1": attack3,
                                     'change': f"{get_difference_emoji(attack_diff2_3)} {abs(attack_diff2_3):.0f}"},
                                    {"attribute": "Defense", "pokemon_1": defense3,
                                     'change': f"{get_difference_emoji(defense_diff2_3)} {abs(defense_diff2_3):.0f}"},
                                    {"attribute": "Speed", "pokemon_1": speed3,
                                     'change': f"{get_difference_emoji(speed_diff2_3)} {abs(speed_diff2_3):.0f}"},
                                    {"attribute": "Sp. Attack", "pokemon_1": spattack3,
                                     'change': f"{get_difference_emoji(spattack_diff2_3)} {abs(spattack_diff2_3):.0f}"},
                                    {"attribute": "Sp. Defense", "pokemon_1": spdefense3,
                                     'change': f"{get_difference_emoji(spdefense_diff2_3)} {abs(spdefense_diff2_3):.0f}"},
                                ]
                            ),
                        ])
                    ])
                ])
        return front_content, shiny_content, attribute_fig, \
            [
                html.P(f'Generation: {gen}',style={'fontSize': 24}),
                html.P(f'Status: {sub_df.status.unique()[0]}',style={'fontSize': 24}),
                html.P(f'Species: {sub_df.species.unique()[0]}',style={'fontSize': 24}),
                html.Div('Type: ',style={'fontSize': 24}),
                html.Div(children=type_bar),
                html.P(),
                html.Div('Gender: ',style={'fontSize': 24}),
                html.Div(children=gender_bar),
                html.P(),
                html.Div('Abilities: ',style={'fontSize': 24}),
                html.Div(children=ability_bar)
            ],evol_tree


if __name__ == '__main__':
    application.run(debug=False, port=8050)
