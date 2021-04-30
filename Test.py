from Boardgame.BoardgamesRecommender import BoardgameRecommender

my_instance = BoardgameRecommender(5000)
#my_instance.scrape_boardgames()
#my_instance.save_as_csv()
#my_instance.save_as_pickle()

my_instance.read_pickle()

my_instance.get_boardgame_attrs(name='Scrawl')
my_instance.get_boardgame_attrs(name='galaxy trucker')

print(my_instance.popular_boardgames(10))
print(my_instance.popular_recs_filtered(10, num_players=5, playing_time=60))