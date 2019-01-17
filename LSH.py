# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 17:06:15 2018

@author: yabinghu
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import collections
import itertools
import csv

#######################################Problem 1 read in this data from a TXT  to a M * N matrix, where M is the number of movies and N is the number of users (who rated less than 20 movies)
df = pd.read_csv('Netflix_data.txt', names=['ID', 'Rating', 'Date'])
df = df.drop('Date', 1)
nan_rows = df[df.isnull().T.any().T]
nan_rows = nan_rows.drop('Rating', 1)
df = df[df['Rating'] >= 3.0]
df = df.drop('Rating', 1)
df['index'] = df.index
S1 = df
df = df['ID'].value_counts()
df = df[df.values <= 20]  # user who rated less than 20 movies
df = df.to_frame()
df['Ture ID'] = df.index
df.columns = ['Count', 'ID']
df = pd.merge(S1, df)
df = df.drop('Count', 1)
df.columns = ['ID', 'Map']
df = df.sort_values(by='Map')
ulist = df.Map.tolist()
mlist = nan_rows.index.tolist()
mdict = pd.Series(nan_rows.ID,
                  index=nan_rows.index)  # series for all the movie ID, its index is the index of its position at original input
udict = pd.Series(df.ID,
                  index=df.index)  # series for all the users ID,its index is the index of its position at original input


udict.index = df.Map
j = 1
S8 = []  # mapping between Movie and User ID, the result is the index for Movies at  original dataset
for i in range(len(ulist)):
    if ulist[i] < mlist[j]:
        S8.append(mlist[j - 1])
    if ulist[i] > mlist[j]:
        S8.append(mlist[j])
        j = j + 1
        if j >= len(mlist) - 1:
            j = len(mlist) - 1

S10 = udict.values.tolist()  # N users IDs
S10 = list(map(int, S10))
S9 = []  # All the Movie ID corresponding to N users IDs
for i in range(len(S8)):
    S9.append(int(mdict[S8[i]].replace(':', '')))
data = pd.DataFrame({'MID': S9, 'UID': S10})  # final result
data=data.sort_values(by=['UID'])
UID = data.UID.tolist()
MID=data.MID.tolist()
matrix = np.zeros((len(data.MID.unique()), len(data.UID.unique())))

data = data['UID'].value_counts()
data = data.to_frame()
data['new_index'] = data.index
data = data.sort_values(by='new_index')
data.columns = ['Count', 'new_index']
count = data.Count.tolist()
user_id_map= data.new_index.tolist()
temp=0
j=0
for i in range(len(MID)):
        matrix[MID[i]-1, j] = 1
        if i==temp+count[j]-1:
            temp=i+1
            j=j+1
            
del df,S1,S10,S8,S9,udict,ulist,data,mdict,mlist,nan_rows,count
#%%
#########Problem 2 Pick 10,000 pairs at random and compute the average Jaccard distance of the pairs, as well as the lowest distance among them then plot
def find_jaccard_dist(matrix):
    rand_pairs = []     
    pair=random.sample(range(0,matrix.shape[1]),20000)
    for i in range(0,20000,2):
        rand_pairs.append([pair[i],pair[i+1]])
    jaccard_dist = []
    for elements in rand_pairs:
        union = np.size(np.nonzero(matrix[:,elements[0]]+matrix[:,elements[1]]) ) 
        intersection = np.dot(matrix[:,elements[0]],matrix[:,elements[1]])
        jaccard_dist.append(1-intersection/union)
    return jaccard_dist

jaccard_dist=find_jaccard_dist(matrix)
average_dist = np.average(jaccard_dist)
min_dist=np.min(jaccard_dist)
print('The average jaccard distance is',average_dist)
print('The minimum jaccard distance is',min_dist)
plt.hist(jaccard_dist,100)
plt.title('Histgram for problem 2')
plt.xlabel('Jaccard distance')
plt.ylabel('the number of pairs')
plt.show()
del jaccard_dist,i,j,temp


#%%

#Problem 3 find the approximate nearest neighbor of a queried user.
def efficient_data_storage():
    eff_list=[]
    for i in range(matrix.shape[1]):
       
        eff_list.append(list(np.where( matrix[:,i]== 1)[0]))
    return eff_list
    
size=matrix.shape[1]
eff_list=efficient_data_storage()
del matrix
#%%
#Problem 4 detect all pairs of users that are close to one another.
def hash_func_permutation(x,a,b,n):
    return (a*x + b) % n

def hash_func_parameter(n,r):
    rand_paras=random.sample(range(1,r),n)
    return np.array(rand_paras)        

def sig_matrix(size,n):
    a=hash_func_parameter(n,4507)
    b=hash_func_parameter(n,4507)
    x = np.arange(4499)
    x=np.repeat(x, n)
    x=x.reshape(4499,n)
    Sig = np.zeros([n,size])
    temp=hash_func_permutation(x,a,b,4507)
    for i in range(n):
        for k in range(size):
            Sig[i][k]=min(temp[eff_list[k],i])
    return Sig

Sig=sig_matrix(size,161)   

def lsh_min_hash(Sig,b1,r):
    result = []
    a=hash_func_parameter(161,4507)
    b=hash_func_parameter(161,4507)
    a=a.reshape(161,1)
    b=b.reshape(161,1)
    Sig_2 = hash_func_permutation(Sig,a,b,4507)
    for k in range(b1):
        band=Sig_2[r*k:r*(k+1)]
        band_f=band.sum(0)
        counter=collections.Counter(band_f) 
        dups=[i for i in counter if counter[i]>1] 
   
        for item in dups:
            result.append(list(np.where( band_f== item)[0]))
    return result

b1=23
r=7
result=lsh_min_hash(Sig,b1,r)

def jaccard_sim(s1, s2):
    s1 = set(s1)
    s2 = set(s2)
    return len(s1.intersection(s2)) / len(s1.union(s2))

def find_pairs(result,eff_list):
    result.sort()
    result=list(result for result,_ in itertools.groupby(result))
    ture_pairs=[]
    for elements in result:
                for pair in itertools.combinations(elements,2):
                    if jaccard_sim(eff_list[pair[0]], eff_list[pair[1]]) >= 0.65:
                        ture_pairs.append((pair[0], pair[1]))
    ture_pairs.sort()
    ture_pairs = list(ture_pairs for ture_pairs,_ in itertools.groupby(ture_pairs))

    return ture_pairs

ture_pairs=find_pairs(result,eff_list)   

with open('similarPairs.csv','w') as writeFile:
  similarWriter = csv.writer(writeFile, delimiter=',')

  for i in range(len(ture_pairs)):
    similarWriter.writerow([ture_pairs[i][0], ture_pairs[i][1]])
#%%
#Problem 5 Create a function that accepts a queried user and returns their approximate nearest neighbor.
def query_neighbor(movie_id,eff_list):
    similarity = np.zeros(231424)
    movie_index=[x-1 for x in movie_id]
    for i in range(len(eff_list)):
        similarity[i]= jaccard_sim(eff_list[i], movie_index)
    max_jac_sim=np.max(similarity)
    nearest_neighbors=list(np.where( similarity== max_jac_sim)[0])
    return nearest_neighbors

nearest_neighbors=query_neighbor(input_list,eff_list)
#new_map=np.array(user_id_map)
#neighbors=new_map[neighbors]        
if len(nearest_neighbors)==0:
    print('No nearest neighbors found!')
else:
    print('The nearest neighbors are:',nearest_neighbors)

