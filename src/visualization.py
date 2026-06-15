"""
visualization.py
----------------
Responsável pelas visualizações e análise exploratória dos dados do experimento da catapulta.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_correlacao(df: pd.DataFrame) -> None:
    """
    Exibe matriz de correlação entre variáveis.
    """

    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matriz de Correlação")
    plt.tight_layout()
    plt.show()

def plot_histogramas(df: pd.DataFrame) -> None:
    """
    Exibe histogramas de todas as variáveis numéricas.
    """

    df.hist(figsize=(12, 8))
    plt.tight_layout()
    plt.show()

def plot_scatterplots(df: pd.DataFrame, features: list[str], target: str) -> None:
    """
    Exibe scatterplots das variáveis de entrada
    contra a variável alvo.
    """

    for feature in features:
        if feature not in df.columns:
            continue

        plt.figure(figsize=(6, 4))
        sns.scatterplot(data=df, x=feature, y=target)
        plt.title(f"{feature} vs {target}")
        plt.tight_layout()
        plt.show()

def plot_boxplots(df: pd.DataFrame, features: list[str], target: str) -> None:
    """
    Exibe boxplots das variáveis categóricas/discretas
    contra a variável alvo.
    """

    for feature in features:
        if feature not in df.columns:
            continue

        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df, x=feature, y=target)
        plt.title(f"Boxplot: {feature} vs {target}")
        plt.tight_layout()
        plt.show()

def plot_interacoes(df: pd.DataFrame, target: str, x_col: str = "firing_angle", hue_col: str  = "release_angle") -> None:
    """
    Visualiza possíveis interações entre fatores.
    """
    if x_col not in df.columns or hue_col not in df.columns:
        return

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x=x_col, y=target, hue=hue_col, s=80)
    plt.title(f"Interação: {x_col} x {hue_col}")
    plt.tight_layout()
    plt.show()

def plot_replicas(df: pd.DataFrame, target: str, replica_col: str = "replica") -> None:
    """
    Analisa variabilidade entre réplicas.
    """
    if replica_col not in df.columns:
        return

    plt.figure(figsize=(6, 4))
    sns.boxplot( data=df, x=replica_col, y=target)
    plt.title("Variabilidade entre Réplicas")
    plt.tight_layout()
    plt.show()

def plot_top_resultados(df: pd.DataFrame, target: str, ensaio_col: str = "ensaio", top_n: int = 10) -> None:
    """
    Exibe os melhores resultados do experimento.
    """
    if ensaio_col not in df.columns:
        return

    top = (df.sort_values(target, ascending=False).head(top_n))

    plt.figure(figsize=(10, 5))
    sns.barplot(data=top, x=ensaio_col, y=target)
    plt.title(f"Top {top_n} Ensaios")
    plt.tight_layout()
    plt.show()

def plot_real_vs_prediction(y_real: np.ndarray, y_pred: np.ndarray, title: str = "Valores Reais vs Preditos") -> None:
    """
    Compara valores reais com a predição realizada pelo modelo.
    """
    min_value = min(y_real.min(), y_pred.min())
    max_value = max(y_real.max(), y_pred.max())

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_real, y=y_pred, s=80)
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="red", label="Predição perfeita")
    plt.xlabel("Valor real")
    plt.ylabel("Valor predito")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_residuos(y_pred: np.ndarray, residuals: np.ndarray, title: str = "Resíduos vs Predições") -> None:
    """
    Mostra os resíduos em função dos valores preditos.
    """
    plt.figure(figsize=(7, 4))
    sns.scatterplot(x=y_pred, y=residuals, s=80)

    plt.axhline(0, linestyle="--", color="red")

    plt.xlabel("Valor predito")
    plt.ylabel("Resíduo")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_distribuicao_residuos(residuals: np.ndarray, title: str = "Distribuição dos Resíduos") -> None:
    """
    Mostra a distribuição dos erros do modelo.
    """
    plt.figure(figsize=(7, 4))
    sns.histplot(residuals, kde=True)

    plt.axvline(0, linestyle="--", color="red")

    plt.xlabel("Resíduo")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_coeficientes(theta: np.ndarray, feature_names: list[str] | np.ndarray, top_n: int = 20, title: str = "Coeficientes do Modelo") -> None:
    """
    Mostra os coeficientes mais relevantes em módulo.
    """
    coef_df = pd.DataFrame({"termo": feature_names, "coeficiente": theta, "abs_coeficiente": np.abs(theta)})
    coef_df = coef_df.sort_values("abs_coeficiente", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=coef_df, x="coeficiente", y="termo")
    plt.axvline(0, linestyle="--", color="black")
    plt.title(title)
    plt.xlabel("Coeficiente")
    plt.ylabel("Termo polinomial")
    plt.tight_layout()
    plt.show()

def plot_valores_singulares(singular_values: np.ndarray, title: str = "Valores Singulares da Matriz Phi") -> None:
    """
    Mostra os valores singulares retornados pelo lstsq.
    Útil para avaliar estabilidade numérica.
    """
    plt.figure(figsize=(7, 4))
    sns.lineplot(x=np.arange(1, len(singular_values) + 1), y=singular_values, marker="o")
    plt.xlabel("Índice")
    plt.ylabel("Valor singular")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_comparacao_metricas(results_df: pd.DataFrame, metric: str, title: str | None = None) -> None:
    """
    Compara uma métrica entre datasets.
    """
    if results_df.empty or metric not in results_df.columns:
        return

    plt.figure(figsize=(9, 5))
    sns.barplot(data=results_df, x="dataset", y=metric)
    plt.title(title or f"Comparação de {metric.upper()}")
    plt.xlabel("Dataset")
    plt.ylabel(metric.upper())
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_exploratory_analysis(df_name: str, df: pd.DataFrame, features: list[str], target: str,) -> None:
    print(f"[PLOT] Análise exploratória de '{df_name}':")

    plot_correlacao(df)
    plot_histogramas(df)
    plot_scatterplots(df, features, target)
    plot_boxplots(df, features, target)
    plot_interacoes(df, target)
    plot_replicas(df, target)
    plot_top_resultados(df, target)