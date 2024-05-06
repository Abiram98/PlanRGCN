#!/usr/bin/env python
# coding: utf-8

# In[2]:


from inductive_query.res_proc_helper import *
import importlib
import inductive_query.utils as ih
importlib.reload(ih)
CompletelyUnseenQueryExtractor = ih.CompletelyUnseenQueryExtractor

path = '/data/DBpedia_3_Full_weight_loss'
split_path = f"{path}/test_sampled.tsv"
#c = CompletelyUnseenQueryExtractor(path)
#q_files = c.run()
#print(len(q_files))


# In[2]:


path = '/data/Wikidata_3_Full_weight_loss'
split_path = f"{path}/test_sampled.tsv"
c1 = CompletelyUnseenQueryExtractor(path)
q1_files = c1.run()
print(len(q1_files))


# In[6]:


import pandas as pd
import os
from pathlib import Path
df = pd.read_csv(split_path, sep='\t')
df['id'] = df['id'].apply(lambda x: x[20:])
#print(df)
files = [Path(x).name for x in q1_files]
com_df = df[df['id'].isin(files)]
com_df['time_cls'] = com_df['mean_latency'].apply(lambda x : 0 if x<1 else 1 if x < 10 else 2)
print(com_df['time_cls'].value_counts())

