import geopandas as gpd
import numpy as np
import pandas as pd
import pydeck as pdk # criar mapa no stremlit
import shapely
import streamlit as st

# bublioteca para carregar o arquivo do modelo de machine learning
from joblib import load

# importando os dados, é necessário passar o caminho do home.py até o arquivo de importação
from notebooks.src.config import DADOS_LIMPOS, DADOS_GEO_MEDIAN, MODELO_FINAL

# criando uma função para carregar os dados limpos na página
@st.cache_data
def carregar_dados_limpos():
    return pd.read_parquet(DADOS_LIMPOS)

# criando uma função para carregar os dados com medianas geográficas
@st.cache_data
def carregar_dados_geo():
    gdf_geo = gpd.read_parquet(DADOS_GEO_MEDIAN)

    # Explodir MultiPolygons em polígonos individuais
    gdf_geo = gdf_geo.explode(ignore_index=True)

    # Função para verificar e corrigir geometrias inválidas
    def fix_and_orient_geometry(geometry):
        if not geometry.is_valid:
            geometry = geometry.buffer(0)  # Corrigir geometria inválida
        # Oriente o polígono no sentido anti-horário se for um polígono ou multipolígono
        if isinstance(
            geometry, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)
        ):
            geometry = shapely.geometry.polygon.orient(geometry, sign=1.0)
        return geometry

    # Aplique a função de correção e orientação às geometrias
    gdf_geo["geometry"] = gdf_geo["geometry"].apply(fix_and_orient_geometry)

    # Extrair coordenadas poligonais
    def get_polygon_coordinates(geometry):
        return (
            [[[x, y] for x, y in geometry.exterior.coords]]
            if isinstance(geometry, shapely.geometry.Polygon)
            else [
                [[x, y] for x, y in polygon.exterior.coords]
                for polygon in geometry.geoms
            ]
        )

    # Aplique a conversão de coordenadas e armazene em uma nova coluna
    gdf_geo["geometry"] = gdf_geo["geometry"].apply(get_polygon_coordinates)

    return gdf_geo

# criando uma função para carregar o modelo
@st.cache_resource
def carregar_modelo():
    return load(MODELO_FINAL)


df = carregar_dados_limpos()
gdf_geo = carregar_dados_geo()
modelo = carregar_modelo()

# incluir título
st.title("Previsão de preços de imóveis")

# pegando os nomes dos condados e transformando em lista
condados = sorted(gdf_geo["name"].unique())

# dividindo a tela do site em containêres (divisões)
coluna1, coluna2 = st.columns(2)

# usando um gerenciador de contexto para colocar o conteúdo da coluna1 nela
with coluna1:

    # fazendo a página ficar submissa ao formulário, só recarrega quando clicar em prever preço
    with st.form(key="formulario"):
        
        # transformando a lista em caixa selecionável
        selecionar_condado = st.selectbox("Condado", condados)
        
        # selcionando a longitude mediana do condado
        longitude = gdf_geo.query("name == @selecionar_condado")["longitude"].values
        latitude = gdf_geo.query("name == @selecionar_condado")["latitude"].values
        
        housing_median_age = st.number_input("Idade do imóvel", value=10, min_value=1, max_value=50)
        
        total_rooms = gdf_geo.query("name == @selecionar_condado")["total_rooms"].values
        total_bedrooms = gdf_geo.query("name == @selecionar_condado")["total_bedrooms"].values
        population = gdf_geo.query("name == @selecionar_condado")["population"].values
        households = gdf_geo.query("name == @selecionar_condado")["households"].values
        
        median_income = st.slider("Renda média (milhares de US$)", min_value=5.0, max_value=100.0, value=45.0, step=5.0)
        
        ocean_proximity = gdf_geo.query("name == @selecionar_condado")["ocean_proximity"].values
        
        # criando os bins de acordo com as categorias criadas
        bins_income = [0, 1.5, 3, 4.5, 6, np.inf]
        # a partir da faixa de valores a categoria será identificado
        median_income_cat = np.digitize(median_income / 10, bins=bins_income)
        
        rooms_per_household = gdf_geo.query("name == @selecionar_condado")["rooms_per_household"].values
        population_per_household = gdf_geo.query("name == @selecionar_condado")["population_per_household"].values
        bedrooms_per_room = gdf_geo.query("name == @selecionar_condado")["bedrooms_per_room"].values
        
        # colunas de entrada do modelo
        entrada_modelo = {
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "median_income": median_income / 10, # dividindo os valores por 10
            "ocean_proximity": ocean_proximity,
            "median_income_cat": median_income_cat,
            "rooms_per_household": rooms_per_household,
            "population_per_household": population_per_household,
            "bedrooms_per_room": bedrooms_per_room,
        }
        
        # DataFrame com as colunas de entrada do modelo
        df_entrada_modelo = pd.DataFrame(entrada_modelo)
        
        # botão de previsão de preço (após isso a página vai ser atualizada)
        botao_previsao = st.form_submit_button("Prever preço")
    
    # verificando se o botão de previsão foi acionado
    if botao_previsao:
        # se sim, faça a previsão do modelo
        preco = modelo.predict(df_entrada_modelo)
        # valor formatado, por exemplo, US$10.000,00
        valor_formatado = f"{preco[0][0]:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        
        st.metric(label="Preço previsto:", value=f"US$ {valor_formatado}")

# usando um gerenciador de contexto para colocar o conteúdo da coluna2 nela
with coluna2:
    # estado inicial do estado
    view_state = pdk.ViewState(
        # transformando o float 32 em float64 com float()
        latitude=float(latitude[0]), # latitude dinâmica
        longitude=float(longitude[0]), # latitude dinâmica
        zoom=5, # zoom inicial
        min_zoom=5, # zoom mínimo
        max_zoom=15, # zoom máximo
    )

    # camada de polygon, coluna geometry do gdf
    polygon_layer = pdk.Layer(
        "PolygonLayer",
        data=gdf_geo[["name", "geometry"]], # gdf e colunas necessárias para vinculação
        get_polygon="geometry", # coluna para pegar os polygons
        get_fill_color=[0, 0, 255, 100], # cor rgb de preenchimento, o quarto valor é a proporção de transparência
        get_line_color=[255, 255, 255], # cor rgb da linha de contorno
        get_line_width=50, # largura da linha
        pickable=True, # deixando a camada selecioável pelo tooltip
        auto_highlight=True, # destaque ao passar o mouse no polygon
    )

    # pegando o condado selecionado
    condado_selecionado = gdf_geo.query("name == @selecionar_condado")
    
    # camada de destaque
    highlight_layer = pdk.Layer(
        "PolygonLayer",
        data=condado_selecionado[["name", "geometry"]], # pegando o nome e geometria do condado selecionado
        get_polygon="geometry", # coluna para pegar o polígono
        get_fill_color=[255, 0, 0, 100], # cor rgb de preenchimento, o quarto valor é a proporção de transparência
        get_line_color=[0, 0, 0], # cor rgb da linha de contorno
        get_line_width=500, 
        pickable=True, # deixando a camada selecioável pelo tooltip
        auto_highlight=True, # destaque ao passar o mouse no polygon
    )

    # interação ao passar o mouse pelo mapa (HTML, CSS, JavaScript)
    tooltip = {
        "html": "<b>Condado:<b> {name}", # variável html name do gdf
        "style": {
            "background": "steelblue",
            "color": "white",
            "fontsize": "10px"
        }, # stilo css
    }

    # mapa final
    mapa = pdk.Deck(
        initial_view_state=view_state, # visualizção inicial
        map_style="light", # estilo do mapa 
        layers=[polygon_layer, highlight_layer], # camadas
        tooltip=tooltip, # adicionando a interatividade do mouse
    )


    # incluindo o mapa na página
    st.pydeck_chart(mapa)
    





