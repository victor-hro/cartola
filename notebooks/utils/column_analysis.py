import pandas as pd
import matplotlib.pyplot as plt

def column_analysis(df, col, methods=['unique']):
    """
    Realiza várias análises sobre uma coluna específica de um DataFrame.

    Parâmetros:
    -----------
    df : pd.DataFrame
        O DataFrame que contém a coluna a ser analisada.
        
    col : str
        O nome da coluna no DataFrame para a qual serão realizadas as análises.
        
    methods : list, opcional
        Lista de métodos de análise a serem aplicados. As opções disponíveis são:
        - 'unique' : Exibe a quantidade de valores únicos e os valores únicos da coluna.
        - 'numeric_nulls' : Converte a coluna para numérico e calcula a quantidade de valores nulos.
        - 'value_counts' : Exibe a contagem de frequências de cada valor da coluna, incluindo valores nulos.
        - 'negative_values' : Conta o número de valores negativos após a conversão para numérico.
        - 'zero_values' : Conta o número de valores zero após a conversão para numérico.
        - 'boxplot' : Gera um gráfico boxplot da coluna, após conversão para numérico.

    Retorno:
    --------
    None
        A função apenas imprime os resultados das análises e gera um gráfico (se aplicável).
    
    Exemplo de uso:
    ---------------
    column_analysis(df, 'coluna_exemplo', methods=['unique', 'numeric_nulls', 'boxplot'])
    """
    
    # Análise de valores únicos
    if 'unique' in methods:
        nunique_values = df[col].nunique()
        unique_values = df[col].unique()
        print('Quantidade de valores únicos:', nunique_values)
        print(unique_values)
        
    # Análise de valores nulos, convertendo para numérico
    if 'numeric_nulls' in methods:
        print('-' * 50)
        nulls = (
            pd.to_numeric(df[col], errors='coerce')  # Correção aqui, usando df[col]
            .isna()
            .sum()
        )
        print('Quantidade de nulos:', nulls)
        
    if 'value_counts' in methods:
        print('-' * 50)
        counts = df[col].value_counts(dropna=False)
        print('Value counts:', counts)
        
    if 'negative_values' in methods:
        print('-' * 50)
        numeric_df = (pd.to_numeric(df[col], errors='coerce'))
        
        qtd_negative = numeric_df[numeric_df < 0].shape[0]
        print('Quantidade de linhas que retornaram com valor negativo:', qtd_negative)
        
    if 'zero_values' in methods:
        print('-' * 50)
        numeric_df = (pd.to_numeric(df[col], errors='coerce'))
        
        qtd_negative = numeric_df[numeric_df == 0].shape[0]
        print('Quantidade de linhas que retornaram zero:', qtd_negative)
        
    if 'boxplot' in methods:
        print('-' * 50)
        numeric_df = pd.to_numeric(df[col], errors='coerce')
        
        print(numeric_df.describe())
        
        plt.figure(figsize=(6, 4))
        plt.boxplot(numeric_df.dropna())  # Usar o numeric_df diretamente
        plt.title(f'Boxplot de {col}')
        plt.show()
        
    return None