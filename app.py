import os
import json
import requests
import plotly.express as px
import numpy as np
import pandas as pd
import boto3
import io
import base64
from io import BytesIO
from PIL import Image

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_loading_spinners as dls

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial import distance

datasets_path = './datasets/pokedex_(Update_05.20).csv'
logo_url = 'https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/25.png'
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

pokemon_df = pd.read_csv(datasets_path)
pokemon_options = [{"label": name, "value": name} for name in pokemon_df.name.unique()]

def decode_base64_image(image_string):
  base64_image = base64.b64decode(image_string)
  buffer = BytesIO(base64_image)
  # print(buffer["generated_images"])
  return Image.open(buffer)

# 0-Ancient style 1-Future style
def get_new_pokemon(prompt, num, type):
    region_name = 'us-east-1'
    if type == 0:
        inputs = "A primitive pokemon " + prompt
    else:
        inputs = "A robotic pokemon " + prompt

    request_body = {
        "inputs": inputs,
        "num_images_per_prompt": num
    }

    runtime = boto3.client('sagemaker-runtime', region_name=region_name)
    response = runtime.invoke_endpoint(EndpointName='huggingface-pytorch-inference-2023-09-01-16-56-57-989',
                                       ContentType='application/json',
                                       Body=json.dumps(request_body))

    output_data = response['Body'].read()
    # print(output_data)
    return output_data

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
generation_plot1.update_layout(title_text='Distribution of Pokémon Introduced over Generations',
                  yaxis=dict(title=''),
                  xaxis=dict(
                    title='Generation',
                    titlefont_size=16,
                    tickfont_size=14,
                ),
                uniformtext_minsize=8,
                yaxis_ticksuffix = "%",
                hovermode="x unified",
                legend_title_text ='Status'
    )

type_gen = pokemon_df.groupby(['generation','type_1']).size().unstack().fillna(0).transpose()
generation_plot2 = px.imshow(type_gen,labels=dict(x='Generation',y = 'Type',color='Count'),
                             color_continuous_scale='Teal')
generation_plot2.update_layout(title_text='Distribution of Pokémon Type over Generations',
                  yaxis=dict(title='',titlefont_size=16,tickfont_size=14),
                  xaxis=dict(
                    title='Generation',
                    titlefont_size=16,
                    tickfont_size=14,
                ),
                uniformtext_minsize=8,
                legend_title_text ='Count'
    )

generation_plot3 = px.box(pokemon_df, x="generation", y='total_points',points='all',
             color='status',#notched=True,
             color_discrete_sequence=px.colors.qualitative.T10,
             custom_data=[pokemon_df['name'],pokemon_df['status']],
             )
generation_plot3.update_layout(title_text='Evolution of Pokémon Total Points over Generations<br>(HP, Attack, Defense, Speed, Sp. Attack, Sp. Defense)',
                  yaxis=dict(title=''),
                  xaxis=dict(
                    title='Generation',
                    titlefont_size=16,
                    tickfont_size=14,
                ),
                uniformtext_minsize=8,
                legend_title_text ='Status'
    )
generation_plot3.update_traces(hovertemplate='Name:%{customdata[0]}<br>Status:%{customdata[1]}<br>Generation:%{x}<br>Total Points:%{y}')

generation_plot4 = px.scatter(pokemon_df,x='attack', y='defense',animation_frame='generation',size='hp',color='type_1',hover_name='name',
                              log_x=True,size_max=60,range_x=[20,150],range_y=[20,250],
                              custom_data=[pokemon_df['name'],pokemon_df['generation'],pokemon_df['type_1']])
generation_plot4.update_layout(title_text='Evolution of Pokémon Attack and Defense over Generations',
                  yaxis=dict(title='Defense',titlefont_size=16,tickfont_size=14),
                  xaxis=dict(title='Attack',titlefont_size=16,tickfont_size=14),
                uniformtext_minsize=8,
                legend_title_text='Type'
    )
generation_plot4.layout.sliders[0].currentvalue.prefix ='Generation: '
generation_plot4.update_traces(hovertemplate='Name:%{customdata[0]}<br>Gen:%{customdata[1]}<br>Type:%{customdata[2]}<br>Attack:%{x}<br>Defense:%{y}')


features =['hp','attack','defense','speed','sp_attack','sp_defense']
stats_df = pokemon_df[features]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(stats_df)
n_clusters=6
kmeans=KMeans(n_clusters=n_clusters,random_state=123).fit(scaled_data)
pokemon_df['cluster']=kmeans.labels_ +1

pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)
pc_df = pd.DataFrame(data=principal_components,columns=['PC1', 'PC2'])
pc_df['cluster'] = kmeans.labels_ +1
pc_df['cluster'] = pc_df['cluster'].astype(str)
pc_df = pc_df.sort_values(by='cluster')
pc_df['Name'] = pokemon_df['name']
cluster_plot = px.scatter(pc_df,x='PC1',y='PC2',color='cluster',title ='Pokémon Clusters based on Stats',
                          color_discrete_sequence=px.colors.qualitative.Vivid,
                          custom_data=[pc_df['Name'],pc_df['cluster']])
cluster_plot.update_layout(height=800,
                  yaxis=dict(title='PC2',titlefont_size=16,tickfont_size=14),
                  xaxis=dict(title='PC1',titlefont_size=16,tickfont_size=14),
                uniformtext_minsize=8,
                legend_title_text='Cluster'
    )
cluster_plot.update_traces(hovertemplate='Name:%{customdata[0]}<br>Cluster:%{customdata[1]}<br>PC1:%{x}<br>PC2:%{y}')
    

def get_top_5_similar(pokemon_name):
    selected_pokemon = pokemon_df[pokemon_df['name'] == pokemon_name]
    same_cluster_pokemon = pokemon_df[pokemon_df['cluster'] == selected_pokemon['cluster'].values[0]]

    distances = []

    for _, row in same_cluster_pokemon.iterrows():
        if row['name'] == pokemon_name:
            continue
        curr_dist = distance.euclidean(selected_pokemon[features].values, row[features])
        distances.append((curr_dist, row['name']))

    top_5 = sorted(distances, key=lambda x: x[0])[:5]
    return [pokemon[1] for pokemon in top_5]


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
                    html.P("In the Pokémon evolution tree, the process of evolution will lead to a significant increase in the various attributes of the evolved Pokémon. "\
                            "This enhancement usually includes increases in basic stats such as HP (health), attack, defense, special attack, special defense, and speed. "\
                            "Exactly how these stats change depends on the specific Pokémon species and evolution stage.",
                        style={"text-align": "left",'font-size':'20'}),
                    html.P("Evolution usually results in higher base stats, which can make the evolved Pokémon more powerful and versatile in battle. "\
                           "Every Pokémon has its own advantages. For example, some are good at attacking, some are better at defending, and some are much faster than others. "\
                            "If you want to achieve a good result in the battle, you should train each Pokémon according to its characteristics. "\
                            "Not all Pokémon with high stats are the most powerful Pokémon. As long as the combination of abilities and attributes are matched, "\
                                "the Pokémon will become much stronger.",
                        style={"text-align": "left",'font-size':'20'}),
                    html.P("Additionally, evolution sometimes changes the type of Pokémon, which has both advantages and disadvantages in terms of weaknesses and resistances. ",
                        style={"text-align": "left",'font-size':'20'}),
                   html.Div(html.Br()),
                   dbc.Row([
                       dls.Hash(
                       html.Div(id='evolution-tree'),color='#435278',speed_multiplier=2,size=30,fullscreen=False)
                   ]),
                   html.Div(html.Br()),
                   dbc.Button('Comparison Across Generation',color='secondary',size='lg',outline=False,
                              style={'font-size':'30px','width':'100%'}),
                   html.Div(html.Br()),
                    html.P("In our series of descriptive graphics, we delve deep into the intricacies of Pokémon across different generations. "\
                           "Through bar plots, heatmaps, box plots, and dynamic animations, we capture the essence of each generation, highlighting the standout Pokémon and identifying overarching trends. "
                           "Whether you're a seasoned trainer or a newcomer to the Pokémon world, our graphical series offers insightful perspectives into the captivating journey of these creatures across various generations. ",
                        style={"text-align": "left",'font-size':'20'}),      
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
                   ]),
                   html.Div(html.Br()),
                   dbc.Row([
                       dbc.Col([
                          dls.Hash(
                           dcc.Graph(id='generation-plot3',figure=generation_plot3),color='#435278',speed_multiplier=2,size=30,fullscreen=False)
                       ]),
                       dbc.Col([
                          dls.Hash(
                           dcc.Graph(id='generation-plot4',figure=generation_plot4),color='#435278',speed_multiplier=2,size=30,fullscreen=False)
                           
                       ])
                   ]),
                   html.Div(html.Br()),
                   dbc.Button('Pokemon Similarity based on Stats',color='secondary',size='lg',outline=False,
                              style={'font-size':'30px','width':'100%'}),
                   html.Div(html.Br()),
                    html.P("In our exploration of the vast Pokémon universe, we've employed K-means clustering and Principal Component Analysis to delve into the inherent similarities shared by these captivating creatures."\
                           " Using the multi-dimensional aspects of each Pokémon, from HP to Speed, we've crafted similarity plots that vividly showcase groupings and affinities. "\
                            "Further, for any given Pokémon, our analysis pinpoints its top five counterparts that bear the most resemblance in attributes. "\
                                "This approach not only provides a fresh perspective on the relations between Pokémon but also uncovers potential strategy insights for trainers seeking complementary team members.",
                        style={"text-align": "left",'font-size':'20'}),   
                   html.Div(html.Br()),
                   dbc.Row([
                       dbc.Col([
                          dls.Hash(
                           dcc.Graph(id='cluster-plot',figure=cluster_plot),color='#435278',speed_multiplier=2,size=30,fullscreen=False)
                       ]),
                       dbc.Col([
                        html.Div([
                            html.P('Select a Pokémon ',style={'width':'30%','fontSize':24,'padding-left':'20px'}),
                                html.Div(
                                    dcc.Dropdown(
                                        id='select-pokemon-sim', options=pokemon_options,
                                        multi=False,
                                        value='Bulbasaur',
                                    ), style={'width': '25%','font-size':24}
                                ),
                        ],style=dict(display='flex')),
                          dls.Hash(
                           dcc.Graph(id='similarity-plot'),color='#435278',speed_multiplier=2,size=30,fullscreen=False)
                           
                       ])
                   ]),
                   html.Div(html.Br()),
                
               ])

tab2 = dbc.Tab(label='Create A Pokémon 🎭 (Upcoming)', tab_id='tab2',
               style=tab_style, activeTabClassName="fw-bold fst-italic",
               children=[
                   html.Div(html.Br()),
                   dbc.Button('Create Your Own Pokémon',color='secondary',size='lg',outline=False,
                              style={'font-size':'30px','width':'100%'}),
                   html.Div(html.Br()),
                    html.P("Who is the next new Pokémon? Will different Pokémons have their offsprings? "\
                           "In Pokémon Violet and Scarlet, Paradox Pokémon appears. What does other Paradox Pokémon look like? Let's see what new Pokémon the deep learning model would generate for us! 🔮 "\
                           "[We are currently working on productionize the results on the hosted environment. We encourage you to check back later for the finalized visuals. 🔜 ]",
                        style={"text-align": "left",'font-size':'20'}),
                   dbc.Row([
                       dbc.Col([
                            html.Div(html.Br()),
                            html.P('Pokémon #1'),
                            dcc.Dropdown(
                                id='select-pokemon-1', options=pokemon_options,
                                multi=False,
                                value='Bulbasaur'),
                            html.Div(html.Br()),
                            dls.Hash(html.Div(id='p1'),color='#435278',speed_multiplier=2,size=30,fullscreen=False)
                        ],width = 5),
                        dbc.Col([
                            html.Div(html.Br()),
                            html.P('Pokémon #2'),
                            dcc.Dropdown(
                                id='select-pokemon-2', options=pokemon_options,
                                multi=False,
                                value='Pikachu'),
                            html.Div(html.Br()),
                            dls.Hash(html.Div(id='p2'),color='#435278',speed_multiplier=2,size=30,fullscreen=False)
                        ],width =5),
                        dbc.Col([
                            html.Div(html.Br()),
                            dbc.Button('Create!',id='button1',color='warning',size='lg',outline=False,
                              style={'font-size':'20px','width':'100%'}),
                        ],align='end')    
                    ]),
                    html.Div(html.Br()),
                    html.P('Your New Pokémon:', style={"text-align": "center",'font-size':'30px'}),
                    dls.Hash(html.Div(id='new-pokemon-plot'),color='#435278',speed_multiplier=2,size=30,fullscreen=False),
                    html.Div(html.Br()),
                    html.Div(html.Br()),
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
        raise PreventUpdate
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

@app.callback(
    Output('p1', 'children'),
    Input('select-pokemon-1', 'value')
)
def update_pokemon_info(pokemon_name):
    if pokemon_name is None:
        raise PreventUpdate
    else:
        sub_df = pokemon_df[pokemon_df.name == pokemon_name]
        pokedex = sub_df.pokedex_number.unique()[0]
        url = f'https://pokeapi.co/api/v2/pokemon/{pokedex}'
        response = requests.get(url)
        if response.status_code == 200:
            pokemon_fig_data = response.json()
            image_url_front = pokemon_fig_data['sprites']['other']['official-artwork']['front_default']
            front_content = html.Div([
                html.H5('Regular',style={"text-align": "center"}),
                html.Img(src=image_url_front, style=image_style)
            ])
        else:
            front_content = html.P('Pokemon Image Not Found', style={'color': '#957DAD', 'fontSize': 24})
    return front_content

@app.callback(
    Output('p2', 'children'),
    Input('select-pokemon-2', 'value')
)
def update_pokemon_info(pokemon_name):
    if pokemon_name is None:
        raise PreventUpdate
    else:
        sub_df = pokemon_df[pokemon_df.name == pokemon_name]
        pokedex = sub_df.pokedex_number.unique()[0]
        url = f'https://pokeapi.co/api/v2/pokemon/{pokedex}'
        response = requests.get(url)
        if response.status_code == 200:
            pokemon_fig_data = response.json()
            image_url_front = pokemon_fig_data['sprites']['other']['official-artwork']['front_default']
            front_content = html.Div([
                html.H5('Regular',style={"text-align": "center"}),
                html.Img(src=image_url_front, style=image_style)
            ])
        else:
            front_content = html.P('Pokemon Image Not Found', style={'color': '#957DAD', 'fontSize': 24})
    return front_content

@app.callback(
    Output('new-pokemon-plot','children'),
    Input('button1','n_clicks'),
    State('select-pokemon-1','value'),
    State('select-pokemon-2','value')
)
def update(clicks,p1,p2):
    if clicks is None:
        raise PreventUpdate
    else:
        pk_images = get_new_pokemon(f'fuse {p1.lower()} and {p2.lower}', 4, 1)
        raw_data = json.loads(pk_images)
        encode_img = decode_base64_image(raw_data["generated_images"][0])
        content = html.Div(html.Img(src=encode_img),
                           className="d-grid gap-2 d-md-flex justify-content-md-center",)
    return content

@app.callback(
    Output('similarity-plot','figure'),
    Input('select-pokemon-sim','value')
)
def update(pokemon):
    if pokemon is None:
        raise PreventUpdate
    else:
        top_5_similar = get_top_5_similar(pokemon)
        names = [pokemon] + top_5_similar
        bar_data = pokemon_df[pokemon_df['name'].isin(names)]
        fig = px.bar(bar_data, x='name', y=features, title=f"{pokemon} and its Top 5 Most Similar Pokémon",
                     color_discrete_sequence=px.colors.qualitative.Prism)
        fig.update_layout(height=700)
        fig.update_layout(yaxis=dict(title='Points'),
                  xaxis=dict(
                    title='Pokémon',
                    titlefont_size=16,
                    tickfont_size=14,
                ),
                uniformtext_minsize=8,
                legend_title_text='Stats'
    )
    return fig



if __name__ == '__main__':
    application.run(debug=False, port=8080)
