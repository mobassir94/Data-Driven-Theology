# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 09:22:11 2022

@author: MOBASSIR
"""

import numpy as np
import pandas as pd

from metrics import dot_product_similarity,pairwise_euclidean_dists

import os
import gdown

# assets folder
url = "https://drive.google.com/drive/folders/1gMDdBPFYcc0ACCVZr5tJ82JoaN-Le0aR?usp=sharing"

id = "1gMDdBPFYcc0ACCVZr5tJ82JoaN-Le0aR"

files = os.listdir('./data_driven_theology/')
if(len(files)<5):
    print("downloading necessary files....")
    gdown.download_folder(id=id, quiet=True, use_cookies=False)    

print("copy done...")
from laser_cpu_ddt import Laser
laser = Laser()


corpus_emb = np.load('./data_driven_theology/mlt_quran_torah_emb.npy')

sim_quran_torah = pd.read_csv('./data_driven_theology/ensemble_preds_most_sim_pairs_quran_and_old_testament.csv')


def Multilingual_Quran_Bible_Search_Engine(query,size=10,language = 'en',metric = 'dot'):

    query_embedding = laser.embed_sentences(query, lang=language)

    if(metric == 'dot'):
        query_embedding = np.squeeze(np.asarray(query_embedding))
        linear_similarities = dot_product_similarity(corpus_emb, query_embedding)
    else:
        linear_similarities = pairwise_euclidean_dists(corpus_emb, query_embedding)
        linear_similarities = np.squeeze(np.asarray(linear_similarities))
        linear_similarities = np.array(linear_similarities, dtype=np.float32)

    if(metric == 'dot'):
        Top_index_doc = linear_similarities.argsort()[:-(size+1):-1]
    else:
        Top_index_doc = linear_similarities.argsort()[:-(size+1):]
        Top_index_doc = Top_index_doc[:size]

  
    linear_similarities.sort()
    find = pd.DataFrame()
    for i,index in enumerate(Top_index_doc):
        find.loc[i,'t_citation'] = str(sim_quran_torah['t_citation'][index])
        find.loc[i,'t_book'] = str(sim_quran_torah['t_book'][index])
        find.loc[i,'t_chapter'] = str(sim_quran_torah['t_chapter'][index])
        find.loc[i,'t_verse'] = str(sim_quran_torah['t_verse'][index]) 
        find.loc[i,'t_text'] = str(sim_quran_torah['t_text'][index]) 
        
        find.loc[i,'q_Name'] = str(sim_quran_torah['q_Name'][index])
        find.loc[i,'q_Surah'] = str(sim_quran_torah['q_Surah'][index])
        find.loc[i,'q_Ayat'] = str(sim_quran_torah['q_Ayat'][index])
        find.loc[i,'q_Verse'] = str(sim_quran_torah['q_Verse'][index])
        find.loc[i,'similarity_score'] = str(sim_quran_torah['similarity_score'][index])

    for j,simScore in enumerate(linear_similarities[:-(size+1):-1]):
        find.loc[j,'Score'] = simScore
  
    return find
    
