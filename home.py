import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st

# bublioteca para carregar o arquivo do modelo de machine learning
from joblib import load

# importando os dados, é necessário passar o caminho do home.py até o arquivo de importação
from notebooks.src.config import DADOS_LIMPOS, DADOS_GEO_MEDIAN, MODELO_FINAL

# criando uma função para carregar os dados limpos na página
def carregar_dados_limpos():
    return pd.read_parquet(DADOS_LIMPOS)

# criando uma função para carregar os dados com medianas geográficas
def carregar_dados_geo():
    return gpd.read_parquet(DADOS_GEO_MEDIAN)

# criando uma função para carregar o modelo
def carregar_modelo():
    return load(MODELO_FINAL)


df = carregar_dados_limpos()
gdf_geo = carregar_dados_geo()
modelo = carregar_modelo()

# incluir título
st.title("Previsão de preços de imóveis")

# pegando os nomes dos condados e transformando em lista
condados = list(gdf_geo["name"].sort_values())

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
df_entrada_modelo = pd.DataFrame(entrada_modelo, index=[0])

# botão de previsão de preço (acionamento em booleano)
botao_previsao = st.button("Prever preço")

# verificando se o botão de previsão foi acionado
if botao_previsao:
    # se sim, faça a previsão do modelo
    preco = modelo.predict(df_entrada_modelo)
    st.write(f"Preço previsto: US$ {preco[0][0]:.2f}")
