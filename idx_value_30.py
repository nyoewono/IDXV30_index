#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:31:08 2020

@author: Nathanael Luira Yoewono
About: Duplicate the IDXV30 index made by IDX (Indonesian Stock Exchange)
Link paper: https://www.idx.co.id/media/8851/panduan-indeks-idxv30-idxg30.pdf
"""

import pandas as pd
import numpy as np
import yfinance as yf


class IDXVAL30:
    
    def __init__(self, path):
        self.path = path
        self.idx80 = pd.read_csv(path)
        self.ticker_idx80 = list(self.idx80['Kode'].copy())
        self.nticker = len(self.ticker_idx80)
        self.factors = ["trailingPE", "priceToBook"]
        self.col_name = ['PE', 'PBV']
       
             
    def _adjust_ticker(self):
        """Adjust the ticker name by adding .JK"""
        
        for each_ticker in range(self.nticker):
            self.ticker_idx80[each_ticker] = self.ticker_idx80[each_ticker]+'.JK'
        
        # set the index query ticker
        self.dic_idx80 = {index: ticker for index, ticker in enumerate(self.ticker_idx80)}
    
    def _query_data(self):
        """Query the tickers on yfinance"""
        
        # set the query string
        query = " ".join(self.ticker_idx80)
        
        # call the api query for the base universe
        self.idx80_obj = yf.Tickers(query)
    
    def _generate_df_factors(self):
        """Take the relevant factors from the queried ticker object"""
        
        # create pandas df to store each ticker with its corresponding factors
        self.selection_df = pd.DataFrame()
        self.selection_df['ticker'] = self.ticker_idx80
        self.selection_df['PE'] = np.zeros(self.nticker)
        self.selection_df['PBV'] = np.zeros(self.nticker)
        
        # get each company's factor
        for each_ticker in range(self.nticker):
            
            print(self.ticker_idx80[each_ticker])
            
            # iterate for each factors
            for i in range(len(self.factors)):
                
                factor_selected = self.factors[i]
                col_name_selected = self.col_name[i]
                
                try:
                    self.selection_df.loc[each_ticker, col_name_selected] = self.idx80_obj.tickers[each_ticker].info[factor_selected]
                except:
                    self.selection_df.loc[each_ticker, col_name_selected] = np.nan
        
        # exclude all companies with no factor information
        self.selection_df.dropna(inplace = True)
        self.selection_df.reset_index(inplace = True)
        self.selection_df.drop(labels='index', axis=1, inplace = True)
        self.nticker = self.selection_df.shape[0]
                    
    def _winsorisation(self, alpha=0.05):
        """Adjust for outliers based on quantile spread"""
        lb = int(alpha*self.nticker)
        ub = int((1-alpha)*self.nticker)
        
        # adjust and remove the outliers values
        for factor in self.col_name:
            win_factor = self.selection_df[factor].copy()
            win_factor = win_factor.sort_values(ascending = False)
            win_factor.iloc[list(range(lb))] = win_factor.iloc[3]
            win_factor.iloc[list(range(ub, self.nticker))] = win_factor.iloc[ub]
            self.selection_df[factor+'_wins'] = win_factor
    
    def _zscore(self):
        """Get the z-score for each factor so that they are comparable"""
        
        self._winsorisation()
        
        # standardise each value using zscore method
        for i in self.col_name:
            mean = np.mean(self.selection_df[i+'_wins'])
            std = np.std(self.selection_df[i+'_wins'])
            self.selection_df[i+'_zscore'] = self.selection_df[i+'_wins'].apply(lambda x: (x-mean)/std)

    def _agg_zscore(self):
        """Aggregate the score"""
        
        self._zscore()
        wins_col = [i for i in self.selection_df.columns if '_zscore' in i]
        self.selection_df['agg_score'] = np.zeros(self.nticker)
        
        # take the average from each factor used
        for i in wins_col:
            self.selection_df['agg_score'] += self.selection_df[i]
        self.selection_df['agg_score'] = self.selection_df['agg_score']/len(wins_col)
    
    def _select_stocks(self, nstocks=30):
        """Get top 30 stocks"""
        self.selected_stocks = self.selection_df.sort_values(by='agg_score').copy()[:30]
        self.selected_stocks.reset_index(inplace = True)
        self.selected_stocks.drop(labels='index', axis=1, inplace = True)
    
    def _get_each_weight(self, limit_weight = 0.15):
        """Calculate the weights for each stock"""
        lst_index_ticker = self.idx80_obj.symbols
        self.selected_stocks['mc'] = np.zeros(self.selected_stocks.shape[0])
        
        # enumerate each top 30 selected stocks
        for index, each_ticker in enumerate(self.selected_stocks['ticker']):
            index_ticker = lst_index_ticker.index(each_ticker)
            mc_i = self._weight(self.idx80_obj.tickers[index_ticker])
            self.selected_stocks.loc[index, 'mc'] = mc_i
        
        # get the weight for each stocks
        tot_mc = self.selected_stocks['mc'].sum()
        self.selected_stocks['mc_weight'] = np.round(self.selected_stocks['mc']/tot_mc, 2)
        
        # adjust the weight
        if self.selected_stocks.loc[self.selected_stocks['mc_weight']>limit_weight].shape[0]>0:
            self.selected_stocks['adj_mc'] = self.selected_stocks['mc'].copy()
            over_weight_stocks = self.selected_stocks.loc[self.selected_stocks['mc_weight']>limit_weight, :]
            non_over_weight_stock_ticker = list(set(list(self.selected_stocks['ticker']))-set(list(over_weight_stocks['ticker'])))
            non_over_weight_stock = self.selected_stocks[self.selected_stocks['ticker'].isin(non_over_weight_stock_ticker)]
            
            # keep adjusting until limit reached for each stock
            while True:
                self._limit_cap(over_weight_stocks, non_over_weight_stock)
                if self.selected_stocks.loc[self.selected_stocks['adj_mc_weight']>limit_weight].shape[0]==0:
                    break
                else:
                    over_weight_stocks = self.selected_stocks.loc[self.selected_stocks['adj_mc_weight']>limit_weight, :]
                    non_over_weight_stock_ticker = list(set(list(self.selected_stocks['ticker']))-set(list(over_weight_stocks['ticker'])))
                    non_over_weight_stock = self.selected_stocks[self.selected_stocks['ticker'].isin(non_over_weight_stock_ticker)]
    
    def _weight(self, ticker):
        """Get the weight of each company"""
        market_cap = ticker.info['marketCap']
        ff_ratio = ticker.info['floatShares']/ticker.info['sharesOutstanding']
        return market_cap*ff_ratio
    
    def _limit_cap(self, over_weight_stocks, non_over_weight_stock, limit_weight = 0.15):
        """Adjust the overweight stocks"""
        n_over = over_weight_stocks.shape[0]
        adjust_rate = n_over*limit_weight/(1-(n_over*limit_weight))
        mc_s = non_over_weight_stock['adj_mc'].sum()*adjust_rate
        
        # iterate to adjust the old over limit mc
        for index, rows in over_weight_stocks.iterrows():
            mc_s = rows[-1]
            new_mc = (1/n_over)*mc_s
            self.selected_stocks.loc[index, 'adj_mc'] = new_mc
        
        tot_adj_mc = self.selected_stocks['adj_mc'].sum()
        self.selected_stocks['adj_mc_weight'] = np.round(self.selected_stocks['adj_mc']/tot_adj_mc, 3)
        
    def generate_index(self):
        """Run the stock picking algorithm"""
        self._adjust_ticker()
        self._query_data()
        self._generate_df_factors()
        self._agg_zscore()
        self._select_stocks()
        self._get_each_weight()
        
        # return the index 
        df_index = self.selected_stocks[['ticker', 'adj_mc_weight']].copy()
        df_index.columns = ['ticker', 'weight']
        return df_index
        
path = 'index/idx80/0820-0121/idx80.csv'
index = IDXVAL30(path)
index_df = index.generate_index()
#%%
import os
index_df.to_csv(os.getcwd()+'/idxv30.csv', index = False)
#%%


        

