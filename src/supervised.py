"""
supervised.py
-------------
Contém a classe Catapulta, sendo a classe principal do projeto, responsável por armazenar os datasets carregados e fornecer métodos para acessar esses dados.

Aprofundando, ela também contém métodos como o modelo supervisionado que será utilizado, função para ajuste de curvas e otimização, exploração breve de todos os dados armazenados, e etc.
"""

import sys
from pathlib import Path

from typing import Any
import pandas as pd
import numpy as np
import src.visualization as viz
import src.optimization as opt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.__config__ import PATHS

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

FEATURES = [
    "release_angle",
    "firing_angle",
    "cup_elevation",
    "pin_elevation",
    "bungee_elevation"
]

TARGET = "distancia"

DATA_DIR = PATHS.data

class Catapulta:
    """Classe principal do projeto, responsável por servir como a pipeline principal para o fluxo de trabalho."""
    def __init__(self):
        self.fonts: list[str] = []
        self.datasets: dict[str, pd.DataFrame] = {}
        # Nossas métricas e predições vão diretamente para o datasets_data com o nome do dataset que foi feito a comparação, só chegar a estrutura abaixo e usar um .get().
        self.datasets_data: dict[str, Any] = {}

        self.features: list[str] = []
        self.target_col: str = ""
        self.degree: int = 2
        self.model: Any = None
        self.visualization: bool = True

    # TODO: Se estiver se sentindo romântico, dá pra fazer (model) como parâmetro e fazer um self.model = model(**hyperparameters)
    # Só é preciso fazer o import dos modelos no __main__ (ou onde for chamar a função) ao invés daqui.
    def build_model(self, **hyperparameters) -> bool:
        try:
            self.features = FEATURES
            self.target_col = TARGET
            self.model = PolynomialFeatures(**hyperparameters)
        except Exception as e:
            print(f"[Erro] Ocorreu uma falha durante a construção do modelo: {e}")
            return False
        
        return True

    def store_dataset(self, name: str, dataframe: pd.DataFrame) -> None:
        if dataframe.empty:
            print(f"[Aviso] O dataset '{name}' está vazio. Ele não será salvo.")
            return
        
        if name in self.datasets:
            print(f"[Aviso] Substituindo dataset '{name}' já existente.")

        self.datasets[name] = dataframe

        if name not in self.fonts:
            self.fonts.append(name)
    
    def store_data(self, name: str, data) -> None:        
        if name in self.datasets_data:
            print(f"[Aviso] Substituindo dados de treinamento para '{name}'.")

        self.datasets_data[name] = data
    
    def get_fonts(self) -> list[str]:
        return self.fonts
    
    def get_dataset(self, nome: str) -> pd.DataFrame | None:
        if nome in self.datasets:
            return self.datasets[nome]
        else:
            print(f"[Aviso] Dataset '{nome}' não encontrado na lista de datasets.")
        return None
    
    def get_X_y(self, name: str, df: pd.DataFrame):
        if TARGET not in df.columns:
            print(f"[Erro] Coluna alvo '{TARGET}' não encontrada no dataframe.")
            return None, None
        
        feature_cols = FEATURES
        missing_cols = [col for col in feature_cols if col not in df.columns]

        if missing_cols:
            print(f"[Erro] Features ausentes no dataframe '{name}': {missing_cols}")
            return None, None
        
        X = df[feature_cols].to_numpy(dtype=float)
        y = df[TARGET].to_numpy(dtype=float)

        return X, y
    
    def drop_empty_datasets(self) -> None:
        empty_fonts = [nome for nome, df in self.datasets.items() if df.empty]
        for nome in empty_fonts:
            print(f"[Aviso] Removendo dataset '{nome}' porque está vazio.")
            del self.datasets[nome]
            self.fonts.remove(nome)
    
    def load_df(self, nome: str) -> pd.DataFrame:
        if nome not in self.datasets:
            print(f"[Erro] Dataset '{nome}' não encontrado.")
            return pd.DataFrame()

        return self.datasets[nome]
    
    def analyze_datasets(self) -> None:
        if not self.datasets:
            print("[Aviso] Nenhum dataset armazenado para análise.")
            return
        
        for nome, df in self.datasets.items():
            if df.empty:
                print(f"[Aviso] O dataset '{nome}' está vazio. Pulando análise.")
                continue
            
            print(f"[Info] Análise do dataset '{nome}':")
            print(f"- Número de linhas: {len(df)}")
            print(f"- Número de colunas: {len(df.columns)}")
            print(f"- Colunas: {list(df.columns)}")
            print(f"- Tipos de dados:\n{df.dtypes}")
            print(f"- Estatísticas descritivas:\n{df.describe(include='all')}")
            print()
    
    @property
    def parameters(self) -> np.ndarray:
        return self.get_parameters()

    def get_parameters(self, type: str ='first') -> np.ndarray:
        if not self.datasets:
            print("[Aviso] Nenhum dataset armazenado. Parâmetros indisponíveis.")
            return np.array([])
        
        match type:
            case 'first':
                # Retorna os parâmetros do primeiro dataset encontrado
                for nome, df in self.datasets.items():
                    if not df.empty:
                        print(f"[Info] Parâmetros do dataset '{nome}': {len(df.columns)} colunas.")
                        return df.columns.values
            
            case 'last':
                # Retorna os parâmetros do último dataset encontrado
                for nome, df in reversed(self.datasets.items()):
                    if not df.empty:
                        print(f"[Info] Parâmetros do dataset '{nome}': {len(df.columns)} colunas.")
                        return df.columns.values
            
            case 'all':
                # Retorna a união de todos os parâmetros de todos os datasets
                all_columns = set()
                for nome, df in self.datasets.items():
                    if not df.empty:
                        all_columns.update(df.columns)

                print(f"[Info] Parâmetros combinados de todos os datasets: {len(all_columns)} colunas.")
                return np.array(list(all_columns))

            case 'intersection':
                # Retorna a interseção dos parâmetros de todos os datasets
                all_columns = [set(df.columns) for df in self.datasets.values() if not df.empty]
                if not all_columns:
                    print("[Aviso] Nenhum dataset válido encontrado. Parâmetros indisponíveis.")
                    return np.array([])

                common_columns = set.intersection(*all_columns)
                print(f"[Info] Parâmetros comuns de todos os datasets: {len(common_columns)} colunas.")
                return np.array(list(common_columns))

            case _:
                print(f"[Aviso] Tipo de parâmetro desconhecido: '{type}'. Use 'first', 'last', 'all' ou 'intersection'.")
                return np.array([])

        print("[Aviso] Nenhum dataset válido encontrado. Parâmetros indisponíveis.")
        return np.array([])
    
    def adjust_model(self, df_name: str) -> None:
        print(f"[Execução] Iniciando configuração de ajustes de modelo...")

        if df_name not in self.datasets:
            print(f"[Erro] Não foi possível carregar o dataset '{df_name}', essa função deve ser chamada após o load/store dos datasets.")
            return None
        
        df = self.datasets[df_name]

        X, y = self.get_X_y(df_name, df)

        if X is None or y is None:
            print(f"[Erro] Execução de ajuste interrompida pela ausência dos valores de treinamento 'X' e 'y'.")
            return None

        X_train, X_test, y_train, y_teste = train_test_split(X, y, test_size=0.15, random_state=42)
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        Phi_train = self.model.fit_transform(X_train_scaled)
        Phi_test = self.model.transform(X_test_scaled)
        polynomial_feature_names = self.model.get_feature_names_out(FEATURES)

        theta, residuals, rank, singular_values = np.linalg.lstsq(Phi_train, y_train, rcond=None)

        data = {
            "feature_cols": self.features,
            "polynomial_feature_names": polynomial_feature_names,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_teste,
            "scaler": scaler,
            "poly": self.model,
            "phi_train": Phi_train,
            "phi_test": Phi_test,
            "theta": theta,
            "residuals": residuals,
            "rank": rank,
            "singular_values": singular_values,
            "is_adjusted": True
        }

        self.store_data(df_name, data)

        print(
            f"\n[Ajuste] O modelo polinômial para o dataset '{df_name}' está pronto."
            f"\n[Ajuste] Total de features: {len(self.features)} "
            f"| Parâmetros: {Phi_train.shape[1]} "
            f"| Rank: {rank}.\n"
        )
    
    def optimize_model(self, df_name: str, target_distance: float) -> tuple[dict, dict] | dict:
        print(f"[Otimização] Iniciando otimização do modelo...")

        if df_name not in self.datasets:
            print(f"[Erro] Não foi possível carregar o dataset '{df_name}', essa função deve ser chamada após o load/store dos datasets.")
            return {}
        
        data = self.datasets_data[df_name]
        if not data.get("is_adjusted", False):
            print(f"[Erro] O modelo para '{df_name}' ainda não foi ajustado. Execute primeiro a função adjust_model('{df_name}') antes de rodar.")
            return {}
        
        local_result, global_result = opt.compare_optimization_methods(
            target_distance=target_distance,
            scaler=data["scaler"],
            poly=data["poly"],
            theta=data["theta"],
            bounds=opt.BOUNDS,
        )

        data["optimization"] = {
            "target_distance": target_distance,
            "local": local_result,
            "global": global_result,
        }

        best_result = min(
            [local_result, global_result],
            key=lambda result: result["absolute_error"]
        )

        fixed_x = np.asarray(best_result["x"], dtype=float)

        print(f"[Otimização] Resultado local:")
        self.print_optimization_result(local_result)

        print(f"[Otimização] Resultado global:")
        self.print_optimization_result(global_result)

        if self.visualization:
            opt.plot_2d_cut(
                target_distance=target_distance,
                scaler=data["scaler"],
                poly=data["poly"],
                theta=data["theta"],
                bounds=opt.BOUNDS,
                fixed_x=fixed_x,
            )

            opt.plot_3d_surface(
                scaler=data["scaler"],
                poly=data["poly"],
                theta=data["theta"],
                bounds=opt.BOUNDS,
                fixed_x=fixed_x,
                highlight_x=local_result["x"],
                highlight_predicted_distance=local_result["predicted_distance"],
                target_distance=target_distance
            )
        
        return local_result, global_result

    def run_model(self, df_name: str) -> dict:
        print(f"[Execução] Executando modelo...")

        if df_name not in self.datasets:
            print(f"[Erro] Dataset '{df_name}' não foi carregado nem salvo.")
            return {}
        
        data = self.datasets_data[df_name]
        if not data.get("is_adjusted", False):
            print(f"[Erro] O modelo para '{df_name}' ainda não foi ajustado. Execute primeiro a função adjust_model('{df_name}') antes de rodar.")
            return {}
        
        Phi_test = data["phi_test"]
        y_test = data["y_test"]
        theta = data["theta"]

        y_pred = Phi_test @ theta

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            "dataset": df_name,
            "degree": self.model.get_params().get("degree", 2),
            "n_features": len(data["feature_cols"]),
            "n_params": data["phi_train"].shape[1],
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }

        data["y_pred"] = y_pred
        data["metrics"] = metrics

        print(
            f"[Execução] Resultado para '{df_name}':"
            f"| R²={r2:.4f} "
            f"| MAE={mae:.4f} "
            f"| RMSE={rmse:.4f}"
        )

        if self.visualization:
            self.plot_model_analysis(df_name)
        
        return metrics
    
    def run_all(self) -> pd.DataFrame:
        all_metrics = []

        for df_name in self.datasets.keys():
            metrics = self.run_model(df_name)
            all_metrics.append(metrics)
        
        results_df = pd.DataFrame(all_metrics)
        
        if self.visualization:
            self.plot_results_comparison(results_df)
        
        return results_df
    
    def plot_model_analysis(self, dataset_name: str) -> None:
        import src.visualization as viz

        data = self.datasets_data[dataset_name]

        print(f"[PLOT] Análise do modelo: {dataset_name}")

        viz.plot_real_vs_prediction(y_real=data["y_test"], y_pred=data["y_pred"], title=f"Valores Reais vs Preditos - {dataset_name}")
        viz.plot_residuos(y_pred=data["y_pred"], residuals=data["residuals"], title=f"Resíduos vs Predições - {dataset_name}")
        viz.plot_distribuicao_residuos(residuals=data["residuals"], title=f"Distribuição dos Resíduos - {dataset_name}")
        viz.plot_coeficientes(theta=data["theta"], feature_names=data["polynomial_feature_names"], top_n=20, title=f"Top Coeficientes - {dataset_name}")
        viz.plot_valores_singulares(singular_values=data["singular_values"], title=f"Valores Singulares - {dataset_name}")
    
    def plot_results_comparison(self, results_df: pd.DataFrame) -> None:
        import src.visualization as viz

        print("[PLOT] Comparação final entre datasets")

        viz.plot_comparacao_metricas(results_df, metric="rmse", title="Comparação de RMSE")
        viz.plot_comparacao_metricas(results_df, metric="mae", title="Comparação de MAE")
        viz.plot_comparacao_metricas(results_df, metric="r2", title="Comparação de R²")
    
    def print_optimization_result(self, result: dict):
        feature_names = self.features

        print(f"             Método: {result['method']}")
        print(f"             Sucesso: {result['success']}")
        print(f"             Distância alvo: {result['target_distance']:.4f}")
        print(f"             Distância prevista: {result['predicted_distance']:.4f}")
        print(f"             Erro absoluto: {result['absolute_error']:.4f}")

        print("             Configuração encontrada:")

        for name, value in zip(feature_names, result["x"]):
            print(f"                {name}: {value:.4f}")
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"target_col={self.target_col!r}, "
            f"feature_cols={self.features!r}, "
            f"degree={self.degree}, "
            f"visualization={self.visualization}, "
            f"datasets_loaded={len(self.datasets)}, "
            f"datasets={self.fonts}, "
            f"datasets_data={self.datasets_data}, "
            f"model={self.model}"
            f")"
    )

    def __str__(self) -> str:
        dataset_names = list(self.datasets.keys())

        if dataset_names:
            datasets_text = ", ".join(dataset_names)
        else:
            datasets_text = "nenhum dataset carregado"

        return (
            "PolynomialCurveFitter\n"
            f"- Diretório de dados: {DATA_DIR}\n"
            f"- Variável alvo: {self.target_col}\n"
            f"- Grau polinomial: {self.degree}\n"
            f"- Visualização: {'ativada' if self.visualization else 'desativada'}\n"
            f"- Datasets carregados: {len(self.datasets)}\n"
            f"- Nomes: {datasets_text}"
        )
