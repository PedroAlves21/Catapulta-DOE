# Projeto Catapulta DOE
Um projeto simples em Python utilizando técnicas de aprendizado supervisionado, otimização e ajuste de curvas para estimar a distância do lançamento de uma catapulta.
Feito para a matéria de Programação para I.A: Python, da PUC-SP.

**Requisitos**
- Python 3.x
- Dependências instaladas [```pip install -r requirements.txt```]
- Ambiente virtual [recomendado]

## Estrutura do Repositório

```text
Projeto-Catapulta-DOE/
├── data/
|   ├── catapulta_doe.csv
|   └── catapulta_doe_ordenado.csv
├── notebooks/
|   ├── EDA_catapulta_doe.ipynb
|   ├── notebook.ipynb
|   ├── validation.ipynb
|   └── visualization.ipynb
├── src/
|   ├── __config__.py
|   ├── __init__.py
|   ├── __main__.py
|   ├── data_loader.py
|   ├── optimaztion.py
|   ├── supervised.py
|   └── visualization.py
├── requirements.txt
└── README.md
```

## Setup local

1. **Clone o repositório**
   ```bash
   git clone https://github.com/AndreyFernandes-NP/Projeto-Catapulta-DOE.git
   cd Projeto-Catapulta-DOE
   ```
2. **Crie um virtual environment**
   ```bash
    python -m venv env
    ```
3. **Ative o ambiente virtual**
   ```bash
    env/Scripts/activate.bat
    ```
3. **Instale as dependencias**
   ```bash
   pip install -r requirements.txt
    ```
4. **Execute o projeto**
   ```bash
   python -m src
    ```
****
