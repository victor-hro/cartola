primeiro arquivo: notebooks > data_discovery.ipynb

entendendo a estrutura do dataset e os dados.

sexo: object
uf: object
idade: int
dias: int
pviews: int
visitas: int
tempo_total: float
futebol: float
futebol_internacional: float
futebol_olimpico: float
blog_cartola: float
atletismo: float
ginastica: float
judo: float
natacao: float
basquete: float
handebol: float
volei: float
tenis: float
canoagem: float
saltos_ornamentais: float
tempo_total: float
device: object
home: float
home_olimpiadas: float
cartola_status: object


## Transformação dos Dados:

### Premissas:
A coluna UF será removida do conjunto de dados: Apesar de existirem técnicas para tratamento de dados faltantes, optamos por sua remoção considerando dois fatores principais: o aspecto regional dos dados e o aumento de dimensionalidade que teríamos ao transformá-los para uso em machine learning. Iniciamente iremos começar com uma solução mais simples. Ao remover a UF, eliminamos a complexidade de lidar com diferenças regionais neste momento inicial. Em iterações futuras do projeto, podemos reavaliar esta decisão e incorporar a informação de UF utilizando técnicas mais avançadas de redução de dimensionalidade, caso se prove necessário para melhorar o desempenho do modelo.
