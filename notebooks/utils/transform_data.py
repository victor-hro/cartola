import pandas as pd
import numpy as np

def to_object(df: pd.DataFrame, mapped_cols: dict):
    """Converte colunas do DataFrame para o tipo categórico.

    Args:
        df (pd.DataFrame): DataFrame a ser convertido.
        mapped_cols (dict): Dicionário com as colunas e o tipo 'categorical'.

    Returns:
        pd.DataFrame: DataFrame com as colunas convertidas.
    """
    df_copy = df.copy()
    return df_copy.astype(mapped_cols)

def to_numeric(df: pd.DataFrame, list_cols: list):
    """Converte colunas do DataFrame para numérico.

    Args:
        df (pd.DataFrame): DataFrame a ser convertido.
        list_cols (list): Lista com os nomes das colunas a serem convertidas.

    Returns:
        pd.DataFrame: DataFrame com as colunas convertidas.
    """
    df_copy = df.copy()
    for col in list_cols:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    return df_copy

def convert_sec_to_hour(df, list_cols: list):
    "converte unidade de tempo das colunas: de segundos para horas"
    df_copy = df.copy()
    for col in list_cols:
        df_copy[col] = df_copy[col] / 3600
    return df_copy

def remove_outlier(df, column_name):
    df_copy = df.copy()
    # quartis e intervalo interquartil
    Q1 = df_copy[column_name].quantile(0.25)
    Q3 = df_copy[column_name].quantile(0.75)    
    IQR = Q3 - Q1
    
    # limites
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # filtra o dataframe, mantendo apenas os valores dentro dos limites
    df_result = df_copy[(df_copy[column_name] >= lower_bound) & (df_copy[column_name] <= upper_bound)]
    
    print(f"Removendo outliers da coluna '{column_name}'.")
    return df_result

def remove_list_outliers(df, list_cols):
    df_result = df.copy()
    for col in list_cols:
        # atualiza o df iterativamente removendo outliers
        df_result = remove_outlier(df_result, col)
    return df_result

def criar_faixas_etarias(df, coluna_idade='idade'):
    df = df.copy()
    bins = [0, 17, 25, 35, 45, 60, np.inf]
    labels = ['Até 17 anos', '18-25 anos', '26-35 anos', '36-45 anos', '46-60 anos', '60+ anos']
    
    df['faixa_etaria'] = pd.cut(
        df[coluna_idade], 
        bins=bins, 
        labels=labels, 
        right=False
    )
    
    return df