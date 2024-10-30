import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.metrics import recall_score, precision_score
def barplot_grouped_cartola(df: pd.DataFrame
                            , col: str
                            , xlabel: str=''
                            , ylabel: str=''
                            , title=''
                            , orientation='horizontal'
                            , width=900, height=600
                            , theme=px.colors.qualitative.Alphabet):
    # agrupa pela coluna desejada e faz o value_counts
    analise = (
        df.groupby('cartola_status')[col]
        .value_counts()
        .reset_index()
        .rename(columns={0: 'count'})
    )
    
    # calculando percentuais
    analise['percentual'] = 100 * analise['count'] / analise['count'].sum()
    analise['percentual'] = analise['percentual'].round(2)
    
    # cria texto para os rótulos, contendo qtd (%)
    analise['texto'] = analise.apply(
        lambda x: f"{x['count']:,}\n({x['percentual']}%)", 
        axis=1
    )
    
    # criando barplot
    fig = px.bar(
        analise,
        x= 'cartola_status' if orientation == 'horizontal' else 'count',
        y='count' if orientation == 'horizontal' else 'cartola_status',
        color=col,
        barmode='group',
        title=title,
        labels={
            'cartola_status': ylabel,
            col: xlabel if orientation == 'horizontal' else ylabel,
            'count': ylabel if orientation == 'horizontal' else xlabel
        },
        color_discrete_sequence=theme,
        text='texto'
    )
    
    # alterando o layout
    fig.update_layout(
        width=width,
        height=height,
        bargroupgap=0.1,
        showlegend=True,
        template='plotly_white'
    )
    
    # Ajusta a posição dos rótulos, deixando em auto
    fig.update_traces(textposition='auto', marker_line_color='black', marker_line_width=0.5)
    
    fig.show()
    
    
def barplot_counts_cartola(df: pd.DataFrame,
                           col: str,
                           xlabel: str='',
                           ylabel: str='',
                           title: str='',
                           width=800,
                           height=600,
                           order=None,
                           theme=px.colors.sequential.Viridis):
    
    # value_counts de acordo com a coluna desejada
    df_count = df[col].value_counts().reset_index()
    df_count.columns = [col, 'count']

    # calculo dos percentuais e coluna text contendo "valor (percentual %)"
    df_count['percentual'] = 100 * df_count['count'] / df_count['count'].sum()
    df_count['text'] = df_count.apply(lambda x: f"{x['count']} ({x['percentual']:.2f}%)", axis=1)

    if order:
        df_count[col] = pd.Categorical(df_count[col], categories=order, ordered=True)
        df_count = df_count.sort_values(col)
    
    # criando o histograma
    fig = px.bar(
        df_count,
        y=col,
        x='count',
        width=width,
        height=height,
        title=title,
        color=col,
        color_discrete_sequence=theme if theme else px.colors.sequential.Viridis,
        text='text'
    )

    # remove a colorbar e ajusta automaticamente os rótulos nas barras
    fig.update_traces(textposition='auto', marker_line_color='#ec7b00', marker_line_width=2)

    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template='plotly_white',
        title_x=0.5,
        showlegend=False
    )

    fig.show()



def plot_th_decision(model, X, y, th_atual=0.5):
    # Obter probabilidades do modelo
    y_probas = model.predict_proba(X)[:, 1]

    # Criar thresholds de 0.05 em 0.05
    thresholds = np.arange(0.26, 0.6, 0.02)

    # Calcular precision e recall para cada threshold
    precisions = []
    recalls = []

    for th in thresholds:
        y_pred = (y_probas > th).astype(int)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        precisions.append(precision)
        recalls.append(recall)

    # Criar DataFrame com os resultados
    df_metrics = pd.DataFrame({
        'Threshold': thresholds,
        'Precision': precisions,
        'Recall': recalls
    })

    # Criar gráfico com Plotly
    fig = go.Figure()

    # Adicionar linha de Precision
    fig.add_trace(go.Scatter(
        x=df_metrics['Threshold'],
        y=df_metrics['Precision'],
        name='Precision',
        line=dict(color='blue', width=2)
    ))

    # Adicionar linha de Recall
    fig.add_trace(go.Scatter(
        x=df_metrics['Threshold'],
        y=df_metrics['Recall'],
        name='Recall',
        line=dict(color='red', width=2)
    ))

    # Atualizar layout
    fig.update_layout(
        title='Precision e Recall vs Threshold',
        xaxis_title='Threshold',
        yaxis_title='Score',
        hovermode='x unified',
        yaxis=dict(range=[0, 1])
    )

    # Adicionar linha vertical no threshold atual (0.45)
    fig.add_vline(x=th_atual, line_dash="dash", line_color="gray", annotation_text=f"Threshold Atual ({th_atual})")

    # Mostrar o gráfico
    fig.show()

    # Mostrar os valores em uma tabela
    print("\nValores por Threshold:")
    print(df_metrics.round(3).to_string(index=False))