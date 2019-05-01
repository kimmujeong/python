#https://github.com/Yooonkyung/Movie_recommendation/blob/master/04.%20Movie_recommendation_system.ipynb
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
rating=pd.read_csv("C:\\Users\\thgus\\Downloads\\movie\\ratings.csv")
movies=pd.read_csv("C:\\Users\\thgus\\Downloads\\movie\\movies.csv")
movie=pd.merge(rating,movies,how='inner',on='movieId')
movie=movie[['userId','movieId','rating']]
movie=movie.sort_values(['userId'])
#print(movie[movie.duplicated(['userId','title'],keep=False)])
movie=movie.drop_duplicates(['userId','movieId'])
movie_matrix=movie.set_index(['userId','movieId']).unstack()

def cosine_similarity(data_name):
    from sklearn.metrics.pairwise import cosine_distances
    similarity = 1 - cosine_distances(data_name)    # sklearn은 정의와 반대이므로 1에서 빼준다.
    return similarity

movie_sample=movie.sample(100) #10000까지 어느정도 가능
movie_sample=movie_sample.sort_values(['userId'])
cos_sim = cosine_similarity(movie_sample)    # data set으로 df를 넣음
print(cos_sim.shape)
print(cos_sim)








# df2=rating[['userId','movieId','rating']]
# print(df2.set_index(['userId','movieId']).unstack())
# df2=df[['userId','movieId','rating']]
# df2_matrix=df2.set_index(['userId','movieId']).unstack() #매트릭스 형태로
# #print(df2_matrix)
#
# df2=df2.sample(100) #100개 무작위 샘플링.. cos_sim() 돌아가지 않아서
# def cosine_similarity(data_name):
#     from sklearn.metrics.pairwise import cosine_distances
#     similarity = 1 - cosine_distances(data_name)    # sklearn은 정의와 반대이므로 1에서 빼준다.
#     return similarity
#
# cos_sim = cosine_similarity(df2)
# print(cos_sim)