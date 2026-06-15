import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import get_csv, load_csv
from src.supervised import Catapulta
from src.visualization import plot_exploratory_analysis

if __name__ == "__main__":
    print(f"[Execução] Programa iniciado, configurando pipeline...")

    fontes = get_csv()
    cat_cls = Catapulta()

    if not fontes:
        raise FileNotFoundError("[Erro] Nenhum arquivo CSV encontrado em 'data/'")
    
    print(f"[Execução] Pipeline configurada, {len(fontes)} arquivo(s) CSV encontrado(s) em 'data/'.")
    print(f"[Execução] Construindo modelo principal...")

    cat_cls.build_model(degree=2, include_bias=True)

    for fonte in fontes:
        try:
            print(f"[Execução] Carregando dataset '{fonte['nome']}'...")

            df = load_csv(fonte)
            cat_cls.store_dataset(fonte["nome"], df)
            print(f"[Execução] '{fonte['nome']}' carregado com {len(df)} linhas e {len(df.columns)} colunas.")
            #plot_exploratory_analysis(fonte["nome"], df, cat_cls.features, cat_cls.target_col)

            cat_cls.adjust_model(fonte["nome"])
        except Exception as e:
            print(f"[Erro] Ocorreu uma falha durante o processamento do dataset '{fonte['nome']}': {e}")
    
    print(f"[Execução] Fonte(s) disponível(is): {cat_cls.get_fonts()}")
    print(f"[Execução] Análise dos datasets carregados:")
    print(cat_cls.parameters) # cat_cls.get_parameters() também funciona caso queira pegar de um tipo específico além do 'first' padrão
    cat_cls.analyze_datasets()

    print(f"[Execução] Iniciando pipeline da execução de modelos.")
    results_df = cat_cls.run_all()

    print("[Execução] Comparação final:")
    print(results_df.sort_values("rmse"))

    for fonte in fontes:
        try:
            target_distance = 250

            print(f"[Execução] Encontrando otimização para distância={target_distance} no dataset '{fonte['nome']}'")
            local_result, global_result = cat_cls.optimize_model(df_name=fonte["nome"], target_distance=target_distance)
        except Exception as e:
            print(f"[Erro] Ocorreu uma falha durante a otimização de modelo do dataset '{fonte['nome']}': {e}")
    