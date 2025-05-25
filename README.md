# Projeto de previsão de preços de casas na Califórnia

Este projeto tem como objetivo prever o valor mediano das casas para os distritos da Califórnia usando aprendizado de máquina.

## Principais resultados
O estudo foi dividido em cinco partes principais, cada uma com seu próprio caderno na `notebooks` pasta:
1.**Limpeza de dados e engenharia de recursos**: O conjunto de dados foi limpo e transformado para melhorar o desempenho dos modelos de aprendizado de máquina. [Caderno 1](notebooks/01-jb-limpeza_e_tratamento_de_dados.ipynb).
2.**Análise exploratória de dados**: O conjunto de dados foi analisado para compreender a distribuição das características e da variável-alvo. [Caderno 2](notebooks/02-jb-eda.ipynb).
3.****:. [Caderno 3](notebooks/03-jb-geolocalizacao.ipynb).
4.****:. [Caderno 4](notebooks/04-jb-modelos.ipynb).
5.****:. [Caderno 5](home.py).

## Um pouco mais sobre a base

[Clique aqui](referencias/dicionario_de_dados.md) para ver o dicionário de dados da base utilizada.

## Organização do projeto

```
├── .env               <- Arquivo de variáveis de ambiente (não versionar)
├── .gitignore         <- Arquivos e diretórios a serem ignorados pelo Git
├── ambiente.yml       <- O arquivo de requisitos para reproduzir o ambiente de análise
├── LICENSE            <- Licença de código aberto se uma for escolhida
├── README.md          <- README principal para desenvolvedores que usam este projeto.
|
├── dados              <- Arquivos de dados para o projeto.
|
├── modelos            <- Modelos treinados e serializados, previsões de modelos ou resumos de modelos
|
├── notebooks          <- Cadernos Jupyter. A convenção de nomenclatura é um número (para ordenação),
│                         as iniciais do criador e uma descrição curta separada por `-`, por exemplo
│                         `01-fb-exploracao-inicial-de-dados`.
│
|   └──src             <- Código-fonte para uso neste projeto.
|      │
|      ├── __init__.py  <- Torna um módulo Python
|      ├── config.py    <- Configurações básicas do projeto
|      └── graficos.py  <- Scripts para criar visualizações exploratórias e orientadas a resultados
|
├── referencias        <- Dicionários de dados, manuais e todos os outros materiais explicativos.
|
├── relatorios         <- Dicionários de dados.
│   └── imagens        <- Gráficos e figuras gerados para serem usados em relatórios
```

## Configuração do ambiente

1. Faça o clone do repositório que será criado a partir deste modelo.

    ```bash
    git clone ENDERECO_DO_REPOSITORIO
    ```

2. Crie um ambiente virtual para o seu projeto utilizando o gerenciador de ambientes de sua preferência.

    a. Caso esteja utilizando o `conda`, exporte as dependências do ambiente para o arquivo `ambiente.yml`:

      ```bash
      conda env export > ambiente.yml
      ```

    b. Caso esteja utilizando outro gerenciador de ambientes, exporte as dependências
    para o arquivo `requirements.txt` ou outro formato de sua preferência. Adicione o
    arquivo ao controle de versão, removendo o arquivo `ambiente.yml`.

