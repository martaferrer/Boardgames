from Boardgame.BoardgamesRecommender import BoardgameRecommender
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
import math 
import pandas as pd

my_instance = BoardgameRecommender(5000)
#my_instance.scrape_boardgames()
#my_instance.save_as_csv()
#my_instance.save_as_pickle()

my_instance.read_pickle()


df = my_instance.boardgames_data.copy()

# CHECK NULL
for col in my_instance.boardgames_data.columns:
    print(col, 'has', len(my_instance.boardgames_data[my_instance.boardgames_data[col].isnull() == True]), 'NaN')
    

    #print(col, my_instance.boardgames_data[col].isnull().mean())
#print(my_instance.boardgames_data[my_instance.boardgames_data.isnull()])
print(len(my_instance.boardgames_data[my_instance.boardgames_data.isnull()]))


# CHECK BOARDGAMES TITLE REPETITIONS
# check if there are repetitive games
if(my_instance.boardgames_data['Title'].nunique() != my_instance.boardgames_data.shape[0]):
    print('Mmmh it looks like some playgames are repeted...')
    print(set([x for x in list(my_instance.boardgames_data['Title']) if list(my_instance.boardgames_data['Title']).count(x) > 1]))

    # from here ther are one game with different editions (take the last one?)
    #for x in list(my_instance.boardgames_data['Title']):
        #if list(my_instance.boardgames_data['Title']).count(x) > 1:
            #print(my_instance.boardgames_data[my_instance.boardgames_data['Title'] == x].Year)
    
    # from here ther are one game with different editions (take the last one?)
    print(my_instance.boardgames_data[my_instance.boardgames_data['Title'] == 'Wallenstein'][['Board Game Rank', 'Title', 'Year']])
    print(my_instance.boardgames_data[my_instance.boardgames_data['Title'] == 'Space Hulk'][['Board Game Rank', 'Title', 'Year']])
    print(my_instance.boardgames_data[my_instance.boardgames_data['Title'] == 'Love Letter'][['Board Game Rank', 'Title', 'Year']])

    # For instance we could check if the last editions have more reputation than the old ones.
    # Check the evaluation

# is there any pattern regarding the number of players
#In general, do smaller companies appear to have employees with higher job satisfaction?
#print(df['CompanySize'].value_counts())
print(len(df[df['Num Players Min'].isnull()]))
df.dropna(subset=['Num Players Min'],inplace=True, axis=0)
print(len(df[df['Num Players Min'].isnull()]))

df['Num Players Min'] = pd.to_numeric(df['Num Players Min'])
df['Avg Rating'] = pd.to_numeric(df['Avg Rating'])
#df['Num Players Min'] = df[['Num Players Min', 'Avg Rating']].astype(float)
print(df.groupby(['Num Players Min']).mean()['Avg Rating'])





my_instance.get_boardgame_attrs(name='Scrawl')
my_instance.get_boardgame_attrs(name='galaxy trucker')

print(my_instance.popular_boardgames(10))
print(my_instance.popular_recs_filtered(10, num_players=5, playing_time=60))