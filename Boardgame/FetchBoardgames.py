# Cleaning text exercise

# import statements
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import math
import pickle
import os

class BoardgameRecommender:
    def __init__(self, number=900):
        """ Generic distribution class for calculating and
        visualizing a probability distribution.

        Attributes:
            mean (float) representing the mean value of the distribution
            stdev (float) representing the standard deviation of the distribution
            data_list (list of floats) a list of floats extracted from the data file
            """
        self.boardgames_data = []
        self.number = number
        self.root_path = os.path.dirname(os.path.abspath(__file__))
        #self.scrape_boardgames()

    def scrape_boardgames(self):
        '''

        '''
        # create dataframe
        df_columns = ['Board Game Rank', 'Title', 'Year', 'Description', 'Geek Rating', 'Avg Rating', 'Num Voters', \
            'Num Players Min', 'Num Players Max', 'Best Num Players Min', 'Best Num Players Max', 'Playtime Min', 'Playtime Max', \
            'Player Min Age', 'Language Dependence', 'Weight', 'Category', 'Designer']

        self.boardgames_data = pd.DataFrame(columns=df_columns)

        # there are 100 games per page
        num_pages = math.ceil(self.number/100)
        for page in range(1, num_pages):

            # Step 1: Sending a HTTP request to a URL
            r = requests.get("https://boardgamegeek.com/browse/boardgame/page/"+str(page))

            # Step 2: Parse the html content
            # Use "lxml" rather than "html5lib".
            soup = BeautifulSoup(r.text, "lxml")

            # print title of the webpage
            print(soup.title.text, 'page', page)

            # Step 3: Analyze the HTML tag, where your content lives
            # get the colums of the boargame main table
            boardgame_table = soup.find("table", attrs={"class": "collection_table"})
            boardgame_table_data = boardgame_table.find_all("tr")

            # Get all the headings of Lists
            headings = []
            for column_header in boardgame_table_data[0].find_all("th"):
                # remove any newlines and extra spaces from left and right
                headings.append(column_header.text.replace('\n', ' ').strip())

            # get the content of the table rows
            for row in range(1, len(boardgame_table_data)):
                boardgame = []
                for table, heading in zip(boardgame_table_data[row].find_all("td"), headings):
                    if(heading == headings[0]): # board game rank
                        boardgame.append(table.text.replace('\t', ' ').replace('\n',' ').strip())
                    #elif(heading == headings[1]): # thumbnail image
                    elif(heading == headings[2]): # Title
                        
                        try:
                            # we assume there is the year in parenthesis next to the title
                            boardgame.append(table.text.replace('\t', ' ').replace('\n',' ').split('(')[0].strip())
                            boardgame.append(table.text.replace('\t', ' ').replace('\n',' ').split('(')[1].split(')')[0])
                            boardgame.append(table.text.replace('\t', ' ').replace('\n',' ').split('(')[1].split(')')[1].strip())
                        except:
                            # is there no title? is that too specific?
                            del boardgame[1] # delete the title
                            boardgame.append(table.text.split('\t')[0].replace('\n', ''))
                            boardgame.append(math.nan)
                            boardgame.append(table.text.split('\t')[-3].replace('\n', ''))

                        
                        # get link of the boardgame
                        boargame_link = table.find_all('a', href=True)[0]
                    elif(heading == headings[3]): # Geaek Rating
                        boardgame.append(table.text.replace('\t', ' ').replace('\n',' ').strip())
                    elif(heading == headings[4]): # Avg Rating
                        boardgame.append(table.text.replace('\t', ' ').replace('\n',' ').strip())
                    elif(heading == headings[5]): # Num Voters
                        boardgame.append(table.text.replace('\t', ' ').replace('\n',' ').strip())
                    #elif(heading == headings[6]): # Shop
                    else:
                        continue
                        #print(heading, 'nothing to do')

                # get other characteristics 
                boardgame_charac = self.get_boardgame_specifics(boargame_link.attrs['href'])

                # put it all together
                self.boardgames_data.loc[len(self.boardgames_data)] = boardgame + boardgame_charac

        # get the number of asked boardgames
        self.boardgames_data = self.boardgames_data.head(self.number)

        print(self.boardgames_data.shape)
        print(self.boardgames_data.loc[len(self.boardgames_data)-1])

    def get_boardgame_specifics(self, link):
        '''

        '''
        r = requests.get("https://boardgamegeek.com/"+link)
        game_page = BeautifulSoup(r.text, "html.parser")

        # title of the webpage
        print(game_page.title.text)

        #desc_tag = game_page.find_all('meta',{'property':'og:description'})
        #desc_str = str(desc_tag).split('>',1)[0][16:-27].replace('&amp;ldquo;','"')\
        #    .replace('&amp;rdquo;','"').replace('\n',' ')

        #manipulate js script data and create dictionaries for attribute look up
        script = game_page.find("script", text=re.compile("GEEK.geekitemPreload\s+="))

        # recommended number of players
        try:
            players_min = str(script).split('"minplayers":"')[1].split('",')[0]
            players_max = str(script).split('"maxplayers":"')[1].split('",')[0]
        except:
            print('Number of players not found')
            players_min = math.nan
            players_max = math.nan
            
        # best of user players
        try:
            best_num_players = str(script).split('userplayers":{"best":[')[1].split("]")[0]
            best_num_players_min = best_num_players.split('{"min":')[1].split(',')[0]
        except:
            print('Best number of min players not found ')
            best_num_players_min = math.nan

        try:
            best_num_players = str(script).split('userplayers":{"best":[')[1].split("]")[0]
            best_num_players_max = best_num_players.split('"max":')[1].split('}')[0]  
        except:
            print('Best number of max players not found ')
            best_num_players_max = math.nan

        # playing time
        try:
            playtime_min = str(script).split('"minplaytime":"')[1].split('",')[0]
            playtime_max = str(script).split('"maxplaytime":"')[1].split('",')[0]
        except:
            print('Playtime not found ')
            playtime_min = math.nan
            playtime_max = math.nan

        # player age
        try:
            player_age = str(script).split('"minage":"')[1].split('",')[0]
        except:
            print('Min player age not found ')
            player_age = math.nan

        # language dependence
        try:
            language_dependence = str(script).split('"languagedependence":"')[1].split('",')[0]
        except:
            print('Language dependence not found ')
            language_dependence = math.nan

        # boargame weight (complexity rating)
        try:
            boardgame_weight = str(script).split('"averageweight":')[1].split(',"')[0]
        except:
            print('Complexity weight not found ')
            boardgame_weight = math.nan
            
        # board game designer
        try:
            designer = str(script).split('"boardgamedesigner":[{"name":"')[1].split('","')[0]
        except:
            print('Designer not found')
            designer = math.nan

        # boardgame category
        category_tags = str(script).split('"veryshortprettyname":"')
        cat_list = []
        for index, category in enumerate(category_tags):
            # ignoring the first two indeces
            # index[0] -> <script> tag..
            # index[1] -> Overall -> present in all games
            if (category.split('","')[0] not in cat_list) & (index > 1):
                cat_list.append(category.split('","')[0])
        categories_str = ' | '.join([str(elem) for elem in cat_list])
            
    
        game_characteristics = [players_min, players_max, best_num_players_min, best_num_players_max, \
            playtime_min, playtime_max, player_age, language_dependence, boardgame_weight, categories_str, designer]

        return game_characteristics

    def get_boardgame_attrs(self, name):
        '''
        '''
        try:
            boardgame_details = self.boardgames_data[self.boardgames_data['Title'].str.contains(name, case=False, na=False, regex=False)]
        except:
            print('ERROR - Boardgame {} not found'.format(name))
        
        if(len(boardgame_details) != 0):
            print(boardgame_details)
        else:
            print('ERROR - Boardgame {} not found'.format(name))

        return boardgame_details

    def save_as_csv(self, output_name='data\\boardgame_data.csv'):
        '''
        '''

        file_abs_path = self.root_path + '\\' + output_name

        # Export the data to csv
        self.boardgames_data.to_csv(file_abs_path)
    
    def save_as_pickle(self, output_name = 'data\\boardgame_data.pickle'):
        '''
        '''

        file_abs_path = self.root_path + '\\' + output_name

        # Export the data to pickle
        self.boardgames_data.to_pickle(file_abs_path)
    
    def read_pickle(self, file = 'data\\boardgame_data.pickle'):
        
        file_abs_path = self.root_path + '\\' + file

        # overwrite data
        self.boardgames_data = pd.read_pickle(file_abs_path)

        self.number = self.boardgames_data.shape[0]
        print('Reading boardgame database containing {} entries'.format(self.number))


    def popular_boardgames(self, n_top):
        '''
        INPUT:
        n_top - an integer of the number recommendations you want back
        OUTPUT:
        top_boardgames - a list of the n_top ranked boardgames by title in order best to worst
        '''

        top_boardgames = list(self.boardgames_data['Title'][:n_top])

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
            filtered_boardgames = filtered_boardgames[\
                (filtered_boardgames['Num Players Min'].astype(int) <= num_players) &\
                     (filtered_boardgames['Num Players Max'].astype(int) >= num_players)]

        if playing_time is not None:
            filtered_boardgames = filtered_boardgames[\
                #(filtered_boardgames['Playtime Min'].astype(int) <= playing_time) &\
                    filtered_boardgames['Playtime Max'].astype(int) <= playing_time]

        # create top movies list
        top_boardgames = list(filtered_boardgames['Title'][:n_top])

        return top_boardgames

my_instance = BoardgameRecommender(5000)
my_instance.scrape_boardgames()
my_instance.save_as_csv()
my_instance.save_as_pickle()

my_instance.read_pickle()

my_instance.get_boardgame_attrs(name='Scrawl')
my_instance.get_boardgame_attrs(name='galaxy trucker')

print(my_instance.popular_boardgames(10))
print(my_instance.popular_recs_filtered(10, num_players=5, playing_time=60))

my_instance.boardgames_data