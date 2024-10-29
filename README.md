
# Case Técnico - Victor Hugo Rocha de Oliveira

## Descrição
Este projeto visa realizar uma análise exploratória para identificar os fatores que influenciam a adesão de usuários gratuitos do Cartola FC à versão PRO. Além disso, foi desenvolvido um modelo de propensão utilizando um modelo simples de machine learning.

## Estrutura do Projeto
A estrutura do projeto é organizada da seguinte forma:

```
projeto/
│
├── data/
│   ├── 01_raw/                        # Arquivos brutos (raw)
│   └── 02_handling_outliers/          # Dados após remoção de outliers
│
├── models/                             # Contém o modelo de machine learning treinado
│   └── modelo.joblib                   # Exemplo de arquivo de modelo treinado
│
├── notebooks/                          # Notebooks para análise e modelagem
│   ├── data_discovery.ipynb            # Análise inicial da distribuição dos dados
│   ├── data_pipeline.ipynb             # Limpeza de dados para o modelo de ML
│   ├── analise_exploratoria.ipynb      # Análise exploratória detalhada
│   └── model.ipynb                     # Modelo de machine learning
│
├── notebooks/utils/                    # Pacotes Python utilizados no projeto
│
├── docs/                               # Documentação do projeto
│   ├── case_tecnico.pdf                # Case técnico
│   └── analise_exploratoria.pdf        # Análise exploratória em formato PDF
│
└── README.md                           # Documentação do projeto
```



## Ordem de Visualização do Projeto
1. **Descoberta de Dados**: Abra o notebook `data_discovery.ipynb` para explorar a distribuição dos dados.
2. **Análise Exploratória**: Utilize o notebook `analise_exploratoria.ipynb` para realizar uma análise mais detalhada, incluindo a remoção de outliers.
3. **Pipeline de Dados**: Execute o notebook `data_pipeline.ipynb` para limpar e preparar os dados para modelagem. Esse notebook aproveitou de todas as descobertas, durante a etapa da análise exploratória, para efetuar uma melhor limpeza dos dados.
4. **Modelagem**: Abra o notebook `model.ipynb` para treinar e avaliar o modelo de machine learning.