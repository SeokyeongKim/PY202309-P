import pandas as pd
import urllib.request
import matplotlib.pyplot as plt
import re
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.cluster import KMeans
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import mglearn
from sklearn.cluster import AgglomerativeClustering
import sys
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import random
import copy 
from gensim.models import Doc2Vec
from keras.preprocessing.text import Tokenizer
import numpy as np
from collections import Counter
from wordcloud import WordCloud
import operator
import time
from tqdm import tqdm

# For running time measurements
start_time = time.time()

# Preprocessing of data
print("데이터 전처리 시작!")
print("조금만 기다려주세요. 3분 정도 소요됩니다 :)")
raw_data = pd.read_csv("JAM.csv", engine='python',encoding='CP949')
train_data = pd.DataFrame(raw_data)

train_data = train_data[train_data['가사'].notnull()]

# Remove duplicate lyrics
train_data.shape[0] - train_data['가사'].nunique()
train_data.drop_duplicates(subset=['가사'], inplace=True)
titles = train_data['제목'].reset_index()
groups = train_data['가수'].reset_index()

titles_groups = pd.concat([titles, groups],axis=1)

# Full preprocessed dataframe
preprocessed_train_data = train_data
train_data = train_data['가사'].reset_index()


# Remove lyrics such as emoticons and special characters
train_data['가사'] = train_data['가사'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z ]","")
train_data['가사'].replace('', np.nan, inplace=True)
train_data = train_data.dropna(how = 'any')

# Specify the stop words
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
okt = Okt()

# Use Okt, classify words by morphemes, and perform some level of normalization
X_train = []
for sentence in tqdm(train_data['가사'], desc="토크나이징 진행 중"):
    temp_X = []
    temp_X = okt.morphs(sentence, stem=True)  # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords]  # 불용어 제거
    X_train.append(temp_X)

# Save the tokenized X_test
X_train_nparray = np.array(X_train, dtype='object')
np.save('./X_train',X_train_nparray)

X_train = np.load('./X_train.npy',allow_pickle=True).tolist()

common_texts_and_tags = [
    (text, [preprocessed_train_data['제목'][preprocessed_train_data.index[i]], preprocessed_train_data['가수'][preprocessed_train_data.index[i]]],) for i, text in enumerate(X_train)]
TRAIN_documents = [TaggedDocument(words=text, tags=tags) for text, tags in common_texts_and_tags]

model = Doc2Vec(TRAIN_documents, vector_size=100, window=3, epochs=40, min_count=0, workers=4)
model_name = "doc2vec_100,3,40,0,4"
model.save(model_name)
model = Doc2Vec.load(model_name)



# Assuming you have already trained a Doc2Vec model and stored it in the variable 'model'
# If not, you need to train a Doc2Vec model before using it.
tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_train_data['가수'])
X_singer = tokenizer.texts_to_sequences(preprocessed_train_data['가수'])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_train_data['장르'])
X_genre = tokenizer.texts_to_sequences(preprocessed_train_data['장르'])

# Assuming 'model' is your trained Doc2Vec model
X2 = [] 
for n, x_genre in enumerate(X_genre):
    temp = []

    temp.append(x_genre[0])
    temp.append(X_singer[n][0])
    
    X2.append(temp)

X3 = np.array(X2)

#print("K-Means Clustering")


M_KMeans = KMeans(n_clusters=8, random_state=0)
X = X3   # Document vector 
M_KMeans.fit(X)  # Fitting the model

'''
M_KMeans = KMeans(n_clusters=8, random_state=0)
X = model.docvecs.vectors_docs # document vector 전체를 가져옴. 
M_KMeans.fit(X)# fitting 
'''
 
cluster_dict = {i:[] for i in range(0,8)}
for text_tags, label in zip(common_texts_and_tags, M_KMeans.labels_):
    text, tags = text_tags
    cluster_dict[label].append([tags, text])


unique, counts = np.unique(M_KMeans.labels_, return_counts=True)
dict(zip(unique, counts))

gasas = []
cluster_n = [] 
for clster in range(len(cluster_dict)):
    #print('cluster ' + str(clster))
    cluster_n_song = []
    gasa = []
    for song in cluster_dict[clster]:   
        temp1 = list(preprocessed_train_data['제목'][preprocessed_train_data['제목']==song[0][0]].index)
        temp2 = list(preprocessed_train_data['가수'][preprocessed_train_data['가수']==song[0][1]].index)
        
        for title in temp1:
            x = -1
            if title in temp2:
               x = title
               break

        #print(preprocessed_train_data['제목'][x])
        #print(preprocessed_train_data['장르'][x])
        #print(preprocessed_train_data['가수'][x])
        gasa.append(preprocessed_train_data['가사'][x])
        cluster_n_song.append([preprocessed_train_data['장르'][x],preprocessed_train_data['가수'][x],preprocessed_train_data['소속사'][x],int(preprocessed_train_data['년도'][x]/10000),preprocessed_train_data['작곡'][x]])
    cluster_n.append(cluster_n_song)
    gasas.append(gasa)


# print("Agglomerative Clustering")

n_clusters = 10

M_Agglo = AgglomerativeClustering(linkage='ward',
        connectivity=None, n_clusters=n_clusters)

# Assuming 'model' is my trained Doc2Vec model
X = model.docvecs.vectors_docs

result = M_Agglo.fit_predict(X)

unique, counts = np.unique(result, return_counts=True)
dict(zip(unique, counts))

cluster_dict1 = []
for i in range(n_clusters):
    cluster_dict1.append([])
    
for n, i in enumerate(result):
    text, tags = common_texts_and_tags[n]
    cluster_dict1[i].append([tags, text])

gasas = []
cluster_n = [] 
for clster in range(len(cluster_dict1)):
    #print('cluster ' + str(clster))
    cluster_n_song = []
    gasa = []
    for song in cluster_dict1[clster]:   
        temp1 = list(preprocessed_train_data['제목'][preprocessed_train_data['제목']==song[0][0]].index)
        temp2 = list(preprocessed_train_data['가수'][preprocessed_train_data['가수']==song[0][1]].index)
        
        for title in temp1:
            x = -1
            if title in temp2:
               x = title
               break

        #print(preprocessed_train_data['제목'][x])
        #print(preprocessed_train_data['장르'][x])
        #print(preprocessed_train_data['가수'][x])
        gasa.append(preprocessed_train_data['가사'][x])
        cluster_n_song.append([preprocessed_train_data['장르'][x],preprocessed_train_data['가수'][x],preprocessed_train_data['소속사'][x],int(preprocessed_train_data['년도'][x]/10000),preprocessed_train_data['작곡'][x]])
    cluster_n.append(cluster_n_song)
    gasas.append(gasa)


sentences_tag_n = []
for n, cluster in tqdm(enumerate(gasas), desc="클러스터링 형태소 분석 진행 중", total=len(gasas)):
    sentences_tag = []
    for sentence in cluster:
        morph = okt.pos(sentence)
        sentences_tag.append(morph)
    sentences_tag_n.append(sentences_tag)

noun_adj_list_n = []
for i in range(len(sentences_tag_n)):
    noun_adj_list = []
    for sentence1 in sentences_tag_n[i]:
        for word, tag in sentence1:
            if tag in ['Noun','Adjective']:
                noun_adj_list.append(word)
    noun_adj_list_n.append(noun_adj_list)

banlist = ['사랑','우리','그대','나','너','내']
for n in (noun_adj_list_n):
    for i,v in enumerate(n):
        if len(v) <2:
            n.pop(i)
            continue
        if v in banlist:
            n.pop(i)
            continue


for n in (noun_adj_list_n):
    counts = Counter(n)
    #print(counts.most_common(10))


for n, cluster in enumerate(cluster_n):
    tags_genre = {}
    tags_singer = {}
    tags_house = {}
    tags_time = {}
    tags_composer = {}

    for tag_n in cluster:
        if tag_n[0] not in tags_genre.keys():
            tags_genre[tag_n[0]] = 1
        else:
            tags_genre[tag_n[0]] += 1
        
        if tag_n[1] not in tags_singer.keys():
            tags_singer[tag_n[1]] = 1
        else:
            tags_singer[tag_n[1]] += 1

        if tag_n[2] not in tags_house.keys():
            tags_house[tag_n[2]] = 1
        else:
            tags_house[tag_n[2]] += 1

        if tag_n[3] not in tags_time.keys():
            tags_time[tag_n[3]] = 1
        else:
            tags_time[tag_n[3]] += 1

        if tag_n[4] not in tags_composer.keys():
            tags_composer[tag_n[4]] = 1
        else:
            tags_composer[tag_n[4]] += 1

cluster_tags = []
cluster_tags.append(['다시 한번','숨은'])
cluster_tags.append(['지나간','옛날','추억의'])
cluster_tags.append(['그리움','아이돌'])
cluster_tags.append(['신나는','메탈'])
cluster_tags.append(['드라마틱','발라드','밤'])
cluster_tags.append(['대중적','아이돌','다양한'])
cluster_tags.append(['2세대','노인돌'])
cluster_tags.append(['1세대','들어본'])
cluster_tags.append(['힙한','감각적인'])
cluster_tags.append(['랩','유영진','이국적인'])

cluster_tags_save = np.array(cluster_tags, dtype = 'object')
cluster_dict1 = np.array(cluster_dict1, dtype = 'object')
np.save('./cluster_tags_save',cluster_tags_save)
np.save('./cluster_dict',cluster_dict1)

sys.setrecursionlimit(10000)

# print("hierarchical Clustering")

X = model.docvecs.vectors_docs

linked = linkage(X, 'ward')

print("데이터 전처리 완료!")

#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################


# Ask the user for a song title.
user_input = input("좋아하는 곡 제목을 입력해주세요(ex.Hello, Blueming) : ")
print(f"입력하신 노래는 {user_input} 입니다.")

input_lyrics = []

try:
    # Search the database for lyrics corresponding to a song title
    input_lyrics.append(preprocessed_train_data['가사'][preprocessed_train_data['제목'] == user_input].values[0])
except:
    # If the song you entered is not in the database..
    print("죄송합니다. 입력한 제목과 정확히 일치하는 곡이 데이터베이스에 없습니다.")
    sys.exit() # Exit the program


input_lyrics_tokenized = []
for i in input_lyrics:
    temp_X = []
    temp_X = okt.morphs(i, stem=True) # Tokenization
    temp_X = [word for word in temp_X if not word in stopwords] # Remove stop words
input_lyrics_tokenized.append(temp_X)

similarity_list = []
for text in input_lyrics_tokenized:
    inferred_v = model.infer_vector(text)
    #print(f"vector of {text}: {inferred_v}")
    
    most_similar_docs = model.docvecs.most_similar([inferred_v], topn=5)
    for index, similarity in most_similar_docs:
        similarity_list.append(index)

similarity_cluster = []

cluster_find = -1
for each_song in similarity_list:
    for n, cluster in cluster_dict.items():
            try:
                if each_song in cluster[0]:
                    cluster_find = n
            except:
                break
    similarity_cluster.append(cluster_find)

cluster_cnt = [0,0,0,0,0,0,0,0,0,0]
for i in similarity_cluster:
    cluster_cnt[i] += 1

max = -1
max_n = -1
for n, i in enumerate(cluster_cnt):
    if i > max:
        max = i
        max_n = n

#print(f"{max_n}번 클러스터에 속한 노래들 중 랜덤으로 추출된 곡을 선택하여 추천합니다. ")

songint = []
for i in range(10):
    num = random.randrange(len(cluster_dict[0]))
    while(num in songint):
        num = random.randrange(len(cluster_dict[0]))
    songint.append(preprocessed_train_data['제목'][num])

print(f"'{user_input}' 노래를 좋아한다면 이 노래도 들어보세요! : {songint}")     

end_time = time.time()
elapsed_time = end_time - start_time

minutes = int(elapsed_time // 60)
seconds = round(elapsed_time % 60, 2)
print(f"실행 시간: {minutes}분 {seconds}초")
