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

# variáveis para incluir os dados das colunas no site
longitude = st.number_input("Longitude", value=-122.23)
latitude = st.number_input("Longitude", value=37.88)

housing_median_age = st.number_input("Idade do imóvel", value=10)

total_rooms = st.number_input("Total de quartos", value=800)
total_bedrooms = st.number_input("Total de cômodos", value=100)
population = st.number_input("População", value=300)
households = st.number_input("Domicílios", value=100)

median_income = st.slider("Renda média (múltiplos de US$10.000)", min_value=0.5, max_value=15.5, value=4.5, step=0.5)

ocean_proximity = st.selectbox("Proximidade do oceano", options=df["ocean_proximity"].unique())

median_income_cat = st.number_input("Categoria de renda", value=4)

rooms_per_household = st.number_input("Quartos por domicílio", value=7)
population_per_household = st.number_input("Pessoas por domicílio", value=2)
bedrooms_per_room = st.number_input("Razão de quartos por cômodo", value=0.2)

# colunas de entrada do modelo
entrada_modelo = {
    "longitude": longitude,
    "latitude": latitude,
    "housing_median_age": housing_median_age,
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "population": population,
    "households": households,
    "median_income": median_income,
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
