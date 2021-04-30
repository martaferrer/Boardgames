from .FetchBoardgames import BoardgameScrape

# import statements
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import math
import pickle
import os

class BoardgameRecommender(BoardgameScrape):
    def __init__(self, number=900):
        """ Boardgames recommender class. It inheritates the funcitonalities of 
        BoardgameScrape class. In addition, it provides the methods to:
            - get the ranked-based recommendation
            - get the filtered-based recommendation
        """

        BoardgameScrape.__init__(self, number=number)


    def popular_boardgames(self, n_top):
        '''
        INPUT:
        n_top - an integer of the number recommendations you want back
        OUTPUT:
        top_boardgames - a list of the n_top ranked boardgames by title in order best to worst
        '''

        top_boardgames = list(self.boardgames_data['Title'][:n_top])
        top_boardgames = self.boardgames_data[['Board Game Rank','Title']][:n_top]

        return top_boardgames  # a list of the n_top movies as recommended

    def popular_recs_filtered(self, n_top, num_players=None, playing_time=None, genres=None):
        '''
        REDO THIS DOC STRING

        INPUT:
        n_top - an integer of the number recommendations you want back
        //ranked_movies - a pandas dataframe of the already ranked movies based on avg rating, count, and time
        num_players - a list of strings with years of movies
        playing_time - an integer of the approximate game duration
        genres - a list of strings with genres of boardgames

        OUTPUT:
        top_movies - a list of the n_top recommended boardgames by title in order best to worst
        '''

        filtered_boardgames = self.boardgames_data.copy(deep=True)

        # Filter movies based on year and genre
        if num_players is not None:
            # remove those rows that have nan as num of players
            filtered_boardgames = filtered_boardgames[filtered_boardgames['Num Players Min'].notnull()]
            filtered_boardgames = filtered_boardgames[filtered_boardgames['Num Players Max'].notnull()]

            filtered_boardgames = filtered_boardgames[\
                    (filtered_boardgames['Num Players Min'].astype(int) <= num_players) &\
                     (filtered_boardgames['Num Players Max'].astype(int) >= num_players)]

        if playing_time is not None:
            # remove those rows that have nan as num of players
            filtered_boardgames = filtered_boardgames[filtered_boardgames['Playtime Max'].notnull()]

            filtered_boardgames = filtered_boardgames[\
                #(filtered_boardgames['Playtime Min'].astype(int) <= playing_time) &\
                    filtered_boardgames['Playtime Max'].astype(int) <= playing_time]

        # create top movies list
        #top_boardgames = list(filtered_boardgames['Title'][:n_top])
        top_boardgames = filtered_boardgames[['Board Game Rank', 'Title']][:n_top]

        return top_boardgames

