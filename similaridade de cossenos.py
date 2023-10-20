# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 17:09:37 2023

@author: jurod
"""
#similaridade dos cossenos
import pandas as pd
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#carregar o banco de dados
banco_dados = pd.read_excel(r"C:\Users\jurod\OneDrive - Fatec Centro Paula Souza\Documents\FATEC\2º ciclo\álgebra linear\rotten_tomatoes_dataset.xlsx")

#remover as "stopwords"
vectorizer = TfidfVectorizer(stop_words=('english'))

#selecionar as colunas relevantes para checar a similaridade
dados = banco_dados[['title', 'synopsis']]

#transformar os dados em um único formato de texto
texto = dados.apply(lambda x: ''.join(x.astype(str)), axis=1)

#criar um objeto TF-IDF Vectorizer
vetorizador = TfidfVectorizer()
tfidf_matrix = vetorizador.fit_transform(texto)

#filme escolhido pelo usuario
filme = input('Enter the synopsis of your favorite movie: ').split()

#tranformar o filme em vetor numerico
filme_vetor = vetorizador.transform([' '.join(map(str, filme + [''] * (len(dados.columns) - len(filme))))])

#calcular a similaridade do cosseno entre o filme escolhido pelo usuário e e os filmes do banco de dados
similaridade = cosine_similarity(filme_vetor,tfidf_matrix)

#obter o indice em ordem crescente
indices_similares = similaridade.argsort()[0][::-1]

#selecionar os 5 filmes mais similares
top5_similares = []
for j in indices_similares:
    if banco_dados.loc[j, 'title'] != filme:
        top5_similares.append(j)
    if len(top5_similares) >= 5:
        break

#calcular a similaridade dos filmes com o filme escolhido pelo usuario
dados['Cosine Similarity'] = similaridade[0]
dados['Cosine Angle'] = [math.degrees(math.acos(similarity)) for similarity in similaridade[0]]

#mostrar os 5 filmes recomendados
print('\n')
print('The top 5 movies most similar to your favorite movie are: ')
print(dados.loc[top5_similares, ['title', 'synopsis', 'Cosine Similarity', 'Cosine Angle']])






