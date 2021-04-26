# Cleaning text exercise

# import statements
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import math
import pickle

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
        #self.scrape_boardgames()

    def scrape_boardgames(self):
        '''

        '''
        # create dataframe
        df_columns = ['Board Game Rank', 'Title', 'Year', 'Description', 'Geek Rating', 'Avg Rating', 'Num Voters', \
            'Num Players Min', 'Num Players Max', 'Best Num Players Min', 'Best Num Players Max', 'Playtime min', 'Playtime May', \
            'Player Min Age', 'Language Dependence', 'Weight', 'Designer']

        self.boardgames_data = pd.DataFrame(columns=df_columns)

        for page in range(1, 11):

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
                        boardgame.append(table.text.replace('\t', ' ').replace('\n',' ').split('(')[0].strip())
                        boardgame.append(table.text.replace('\t', ' ').replace('\n',' ').split('(')[1].split(')')[0])
                        boardgame.append(table.text.replace('\t', ' ').replace('\n',' ').split('(')[1].split(')')[1].strip())
                        # get link of the boardgame
                        boargame_link = table.find_all('a', href=True)[0]
                        print(boargame_link.attrs['href'])
                    elif(heading == headings[3]): # Geek Rating
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

        desc_tag = game_page.find_all('meta',{'property':'og:description'})
        desc_str = str(desc_tag).split('>',1)[0][16:-27].replace('&amp;ldquo;','"')\
            .replace('&amp;rdquo;','"').replace('\n',' ')
        print(desc_str)

        #manipulate js script data and create dictionaries for attribute look up
        script = game_page.find("script", text=re.compile("GEEK.geekitemPreload\s+="))

        # recommended number of players
        players_min = str(script).split('"minplayers":"')[1].split('",')[0]
        players_max = str(script).split('"maxplayers":"')[1].split('",')[0]
    
        # best of user players
        try:
            best_num_players = str(script).split('userplayers":{"best":[')[1].split("]")[0]
            best_num_players_min = best_num_players.split('{"min":')[1].split(',')[0]
        except:
            print('Best number of players not found ')
            best_num_players_min = math.nan

        try:
            best_num_players = str(script).split('userplayers":{"best":[')[1].split("]")[0]
            best_num_players_max = best_num_players.split('"max":')[1].split('}')[0]  
        except:
            print('Best number of players not found ')
            best_num_players_max = math.nan

        # playing time
        playtime_min = str(script).split('"minplaytime":"')[1].split('",')[0]
        playtime_max = str(script).split('"maxplaytime":"')[1].split('",')[0]

        # player age
        player_age = str(script).split('"minage":"')[1].split('",')[0]

        # language dependence
        language_dependence = str(script).split('"languagedependence":"')[1].split('",')[0]

        # boargame weight (complexity rating)
        boardgame_weight = str(script).split('"averageweight":')[1].split(',"')[0]

        # board game designer
        try:
            designer = str(script).split('"boardgamedesigner":[{"name":"')[1].split('","')[0]
        except:
            print('Designer not found')
            designer = math.nan
    
        game_characteristics = [players_min, players_max, best_num_players_min, best_num_players_max, \
            playtime_min, playtime_max, player_age, language_dependence, boardgame_weight, designer]

        return game_characteristics

    def get_attributes(self, name):

        try:
            self.boardgames_data[self.boardgames_data['Title'] == name]
        except:
            print('ERROR - Boardgame {} not found'.format(name))

    def save_as_csv(self, output_name='boardgame_data.csv'):
        '''
        '''

        # Export the data to csv
        self.boardgames_data.to_csv(output_name)
    
    def save_as_pickle(self, output_name = 'boardgame_data.pickle'):
        '''
        '''

        # Export the data to pickle
        self.boardgames_data.to_pickle(output_name)


my_instance = BoardgameRecommender()
my_instance.scrape_boardgames()
my_instance.get_attributes(name='Scrawl')
my_instance.save_as_csv()
my_instance.save_as_pickle()
my_instance.boardgames_data