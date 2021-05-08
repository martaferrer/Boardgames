# Boardgames

This is the Capstopne project of the Data Science Nanodegree program offered by Udacity. This project is a gathering of some of the topics learned during this course. 
The goal of this repository is to create a Boardgames dataframe from a url, clean and visualize data, and finally to apply some machine learning algorithms to predict the average rating of a game depending on its features. In addition, it contains some data analysis and visualization. 

## Project components

It is devided into the following chapters:

### 1. Extract data
During this chapter we created the dataset. The key points are:
* Website scraping using BeautifulSoup and request libraries to extract information from the BoardgamesGeeks webpage.
* Use object-oriented programming code by creating classes and inherintance classes
* Structure the repository as to create a Python package
* Accurate documentation
### 2. Transform data
During this chapter the previously obtained data has been assessed and cleaned.
* Clean data: find missing and repetitive values 
* Deal with categorical data
* Data visualization
### 3. Boardgames recommendations
Taking into account the boardgames rank implement a recommender to a speficic user.
* Rank-based recommender
* Knowledge-based recommender
* TODO: Collaborative filtering (user account needed)
### 4. Game rating preditor:
Within this chapter we implemented a boardgame rating predictor from some of the boardgames main features such as complexity, number of players, playing time, among others. 
The predictor is based on a regression machine learning algorithm including:
#### 4.1. Multiple linear regression
#### 4.2. Decision trees
#### 4.3. Random forest regressor
After running the script and output text file will be created with the outcomes from the models.
### 5. Explain your findings with others
A Medium blog is available where all the findins are explained. 

## Project structure

The files in the project follow the following structure:

* Boardgame
  * data
    * boardgame_data.pickle - boardgames database
    * boardgame_data_clean.pkl - output database containing the clean data
  * `FetchBoardgames.py` - BoardGamesGeek Web scraper
  * `BoardgamesRecommender.py` - Top-ranked and knowledge-based boardgames recommender
  * `__init__`
* `boardgames_data_analysis.ipynb` - data cleaning and visualization
* `BoardgamesPredictor.py` - boardgames ratings predictor
* `Test.py` - various
* README.md


## Installation
Th files contains python and jupyter notebook files. It requires Python version 3.* and the following packages: pandas, numpy,
pickle, re, nltk, sklearn

## Acknowledgements
This project is part of the [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025).
