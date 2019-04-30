import pandas as pd
import numpy as np
df=pd.read_csv("C:\\Users\\thgus\\Downloads\\movie\\ratings.csv")
df2=df[['userId','movieId','rating']]
df2_matrix=df2.set_index(['userId','movieId']).unstack() #매트릭스 형태로
#print(df2_matrix)

df2=df2.sample(100) #100개 무작위 샘플링.. cos_sim() 돌아가지 않아서
def cosine_similarity(data_name):
    from sklearn.metrics.pairwise import cosine_distances
    similarity = 1 - cosine_distances(data_name)    # sklearn은 정의와 반대이므로 1에서 빼준다.
    return similarity

cos_sim = cosine_similarity(df2)
print(cos_sim)