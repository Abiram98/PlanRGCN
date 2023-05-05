import numpy as np
import os, sys
import pandas as pd
import matplotlib.pyplot as plt


def bxplot_w_info(data, y_label, save= None, y_range = None, scale='log'):
    plt.clf()
   
    fig, ax = plt.subplots()
    """boxes = [
        {
            'label' : "Male height",
            'whislo': 162.6,    # Bottom whisker position
            'q1'    : 170.2,    # First quartile (25th percentile)
            'med'   : 175.7,    # Median         (50th percentile)
            'q3'    : 180.4,    # Third quartile (75th percentile)
            'whishi': 187.8,    # Top whisker position
            'fliers': []        # Outliers
        }
    
    ]"""
    plt.yscale(scale)
    plt.subplots_adjust(wspace=0,hspace=0,left=0.31,bottom=0.067,right=0.62,top=0.974)
    ax.bxp(data, showfliers=False)
    ax.set_ylabel(y_label)
    if y_range != None:
        ax.set_ylim(y_range)
        
    #ax.set_ylim(data[0]['whislo'],data[0]['whishi'],auto=True)
    #
    #
    if save == None:
        plt.show()
    else:
        plt.savefig(save)
        plt.close()
# The data dictionary should contain mapping from columns to specific values.
def plot_bar_specific(data:dict, save=None):
    plt.clf()
    
    plt.bar(range(len(data)), list(data.values()), align='center')
    plt.xticks(range(len(data)), list(data.keys()))
    if save == None:
        plt.show()
    else:
        plt.savefig(save)
    
def plot_bar_df(data:pd.DataFrame, value_column, key_column, save=None):
    plt.clf()
    
    #plt.bar(key_column,value_column, data =data)
    
    plt.bar(range(len(data)), list(data[value_column]), align='center')
    plt.xticks(range(len(data)), list(data[key_column]), rotation=30, ha='right')
    if save == None:
        plt.show()
    else:
        plt.savefig(save)
        plt.close()

def plot_latency_bxp(df:pd.DataFrame, text="Dataset", column='duration'):
    #df['duration'] = df['duration']* 1000 #ms/s
    mean = df[column].mean()
    q_min =  df[column].min()
    q_25 = df[column].quantile(q=0.25)
    q_50 = df[column].quantile(q=0.5)
    q_75 = df[column].quantile(q=0.75)
    q_100 = df[column].quantile(q=1)
    q_max =  df[column].max()
    print(f"Mean: {mean}, 25% {q_25}, 50% {q_50}, 75% {q_75}, 100% {q_100}")
    #latency_bxplot(df[column], text)
    data = [{
        
            'label' : text,
            'whislo': q_min,    # Bottom whisker position
            'q1'    : q_25,    # First quartile (25th percentile)
            'med'   : q_50,    # Median         (50th percentile)
            'q3'    : q_75,    # Third quartile (75th percentile)
            'whishi': q_max,    # Top whisker position
            'fliers': []        # Outliers
        
        }]
    bxplot_w_info(data, "Query Run times in s")

def plot_operator_freq(df, get_operator_freq):
    skippeable_columns_freq = ['queryID', 'timestamp', 'queryString','tripleCount','duration','resultSize', 'latency','triple', 'bgp',  'treesize','projectVariables', 'leftjoin', 'union', 'joinVertexCount']#'slice',#, 'treesize','projectVariables', 'leftjoin', 'union'
    
    freq = get_operator_freq(df, skipable_cols=skippeable_columns_freq)
    print(freq)
    
    ax = freq.plot(kind='bar',x='operators', y='freq')
    ax.bar_label(ax.containers[0],label_type='edge')
    ax.set_ylabel("Total Operators in dataset")
    plt.yscale('log')
    plt.show()