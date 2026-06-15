# Projeto Catapulta DOE

> Projeto acadêmico desenvolvido na pós-graduação da **PUC-SP**, na disciplina **Programação para IA com Python**, com foco em **modelagem preditiva**, **análise experimental** e **otimização de parâmetros** para estimar a distância de lançamento de uma catapulta.

**Tecnologias e temas aplicados:** `Python` · `Machine Learning` · `DOE` · `Regressão Polinomial` · `Otimização` · `Visualização de Dados`

---

## Visão geral

Este projeto utiliza dados experimentais de uma catapulta para prever a variável de saída **`distancia`** com base em fatores do experimento, como:

- `release_angle`
- `firing_angle`
- `cup_elevation`
- `pin_elevation`
- `bungee_elevation`

A proposta combina conceitos de **Design of Experiments (DOE)** com um pipeline em Python para:

- carregar e organizar os dados experimentais
- ajustar um modelo supervisionado
- comparar métricas de desempenho
- visualizar padrões e erros
- encontrar combinações de parâmetros para atingir uma distância desejada

---

## Objetivo do projeto

Desenvolver uma solução em Python capaz de:

1. **modelar a distância de lançamento** com base nas variáveis do experimento;
2. **avaliar a qualidade do modelo** com métricas estatísticas;
3. **analisar o comportamento dos dados** com gráficos exploratórios;
4. **otimizar os parâmetros da catapulta** para aproximar uma distância-alvo.

---

## Destaques técnicos

- Leitura robusta de arquivos CSV com múltiplos encodings e separadores
- Pipeline orientado a objeto com a classe `Catapulta`
- Transformação polinomial das variáveis de entrada
- Padronização com `StandardScaler`
- Ajuste por mínimos quadrados com `numpy.linalg.lstsq`
- Avaliação por **RMSE**, **MAE** e **R²**
- Comparação entre **otimização local** e **otimização global**
- Visualizações para análise exploratória, resíduos, coeficientes e comparação de resultados

---

## Estrutura do repositório

```text
Projeto-Catapulta-DOE-main/
├── data/
│   ├── catapulta_doe.csv
│   └── catapulta_doe_ordenado.csv
├── notebooks/
│   ├── EDA_catapulta_doe.ipynb
│   ├── notebook.ipynb
│   ├── validation.ipynb
│   └── visualization.ipynb
├── src/
│   ├── __config__.py
│   ├── __init__.py
│   ├── __main__.py
│   ├── data_loader.py
│   ├── optimization.py
│   ├── supervised.py
│   └── visualization.py
├── requirements.txt
├── LICENSE
└── README.md
