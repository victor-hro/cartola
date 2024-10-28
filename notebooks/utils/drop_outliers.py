import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from typing import List, Tuple, Dict

class OutliersProcessor:
    def __init__(self, threshold: float = 3, limite_percentual: float = 5):
        """
        Detecta outliers, faz uma avaliação de acordo com o limite_percentual pré definido,
        definindo se a representatividade das amostras a serem dropadas é menor que 5% ou não
        
        Args:
            threshold: Limite do z-score para considerar um ponto como outlier
            limite_percentual: Percentual limite de outliers para sugerir remoção
        """
        self.threshold = threshold
        self.limite_percentual = limite_percentual
        self.resultados = {}
        
    def process_columns(self
                       , df: pd.DataFrame
                       , colunas: List[str]
                       , plot: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Processa múltiplas colunas para remoção de outliers sequencialmente.
        
        Args:
            df: DataFrame original
            colunas: Lista de colunas para processar
            plot: Se True, gera gráficos para cada coluna processada
            
        Returns:
            Tuple contendo o DataFrame limpo e um dicionário com os resultados
        """
        df_processado = df.copy()
        
        for coluna in colunas:
            # utiliza o zscore do scipy para detecção de outliers
            z_scores = zscore(df_processado[coluna])
            
            # máscara binária entregando informação se o outlier é maior que o limite, padrão = 3 std
            # logo após, filtra no dataframe e armazena na variável outliers
            outliers_mask = np.abs(z_scores) > self.threshold
            outliers = df_processado[outliers_mask]
            
            # verificando representatividade de outliers no df
            total_outliers = len(outliers)
            tamanho_df = len(df_processado)
            percentual_outliers = (total_outliers / tamanho_df) * 100
            
            self.resultados[coluna] = {
                'outliers': outliers,
                'total_outliers': total_outliers,
                'percentual_outliers': percentual_outliers,
                'df_antes': df_processado.copy(),
                'df_depois': df_processado[~outliers_mask].copy()
            }
            
            # removendo outliers
            df_processado = df_processado[~outliers_mask]
            
            print(f"\nProcessando coluna: {coluna}")
            print(f"Tamanho do DataFrame: {tamanho_df}")
            print(f"Quantidade de outliers: {total_outliers}")
            print(f"Percentual de outliers: {percentual_outliers:.2f}%")
            
            if percentual_outliers < self.limite_percentual:
                print("Outliers removidos (menos de 5% dos dados)")
            else:
                print("Alto percentual de outliers detectado")
            
            # Plota gráficos se solicitado
            if plot:
                self._plot_outliers(coluna)
                
        return df_processado, self.resultados
    
    def _plot_outliers(self, coluna: str):
        """Plota histogramas comparando dados com e sem outliers"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Dados originais vs sem outliers
        self.resultados[coluna]['df_antes'][coluna].hist(ax=axes[0], bins=30)
        axes[0].set_title(f'Distribuição Original - {coluna}')
        axes[0].set_xlabel(coluna)
        axes[0].set_ylabel('Frequência')
        
        self.resultados[coluna]['df_depois'][coluna].hist(ax=axes[1], bins=30)
        axes[1].set_title(f'Sem Outliers - {coluna}')
        axes[1].set_xlabel(coluna)
        axes[1].set_ylabel('Frequência')
        
        plt.tight_layout()
        plt.show()