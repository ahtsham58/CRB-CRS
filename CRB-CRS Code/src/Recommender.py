from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import numpy as np
import random
import os

class Recommender:

    def __init__(self):
        self.PATH = os.path.dirname(os.path.abspath(__file__))
        self.ROOT_DIR_PATH = os.path.abspath(os.path.dirname(self.PATH))
        self.ITEM_DATA_path = os.path.join(self.ROOT_DIR_PATH, 'data\\recommenders_item_data\\')
        self.cosine_sim =np.array([])
        self.df_movie_content =pd.DataFrame()
        self.recommended_movies=[]
        self.movies_mentions = pd.DataFrame()
        self.is_session_changed = False
        if self.cosine_sim.size == 0:
            self.cosine_sim, self.df_movie_content = self.data_initialization()

    def data_initialization(self):
        #reading the movies dataset
        movie_ratings_list = pd.read_csv(self.ITEM_DATA_path+ 'movies_rating_data.csv',encoding="Latin1")
        Recommender.movies_mentions = pd.read_csv(self.ITEM_DATA_path+ "movies_data.csv", encoding="Latin1")
        movie_ratings_list = movie_ratings_list.reset_index()
        year_list = []
        for index, row in movie_ratings_list.iterrows():
            title = str(row['title'])
            if title.__contains__('(') and title.__contains__(')'):
                year = int(title[len(title)-5:].replace(')',''))
                year_list.append(year)
            else:
                year = 0000
                year_list.append(year)

        movie_ratings_list['year'] = year_list
        genre_list = ""
        for index,row in movie_ratings_list.iterrows():
                genre_list += row.genres + "|"
        #split the string into a list of values
        genre_list_split = genre_list.split('|')
        #drop-duplicate values
        new_list = list(set(genre_list_split))
        #remove the value that is blank
        new_list.remove('')
        #Enriching the movies dataset by adding the various genres columns.
        movies_with_genres = movie_ratings_list.copy()

        for genre in new_list :
            movies_with_genres[genre] = movies_with_genres.apply(lambda _:int(genre in _.genres), axis = 1)

        #Getting the movies list with only genres like Musical and other such columns
        movie_content_df_temp = movies_with_genres.copy()
        movie_content_df_temp.set_index('databaseId')
        movie_content_df = movie_content_df_temp.drop(columns = ['movieId','rating_mean','title','genres', 'year','databaseId'])
        movie_content_df = movie_content_df.values
        movie_content_df = np.delete(movie_content_df, 0, 1)
        movie_content_df = np.delete(movie_content_df, 0, 1)
        # Compute the cosine similarity matrix
        Recommender.cosine_sim = linear_kernel(movie_content_df,movie_content_df)
        Recommender.df_movie_content = movie_content_df_temp
        return Recommender.cosine_sim, Recommender.df_movie_content
    #Gets the top 10 similar movies based on the content
    def get_similar_movies_based_on_content(self, input_movieId, nb_recom, recommended_movies) :
        try:
            suggestion_list = []
            #cosine_sim, movie_content_df_temp =self.data_initialization(movies_data)
            #create a series of the movie id and title
            movie_content_df_temp = Recommender.df_movie_content.loc[:, ~Recommender.df_movie_content.columns.str.contains('^Unnamed')]
            indicies = pd.Series(movie_content_df_temp.index, movie_content_df_temp['title'])
            title = movie_content_df_temp.loc[movie_content_df_temp['databaseId'] == int(input_movieId)]['title']
            title = title.iloc[0]
            if len(title) < 2:
                return suggestion_list
            movie_index =indicies[title]
            sim_scores = Recommender.cosine_sim[movie_index].tolist()
            df_score = pd.DataFrame(sim_scores,columns=['scores'])
            df_score = df_score.sort_values(by=['scores'], ascending=False)
            df_score = df_score.head(15)
            movie_sim_scores = df_score['scores'].tolist()
            # Get the movie indices
            movie_indices = df_score.index.values.tolist()
            similar_movies = pd.DataFrame(movie_content_df_temp[['title','genres', 'year','rating_mean']].iloc[movie_indices])
            similar_movies = similar_movies[similar_movies.title != title]
            similar_movies = similar_movies.sort_values(by ='rating_mean', ascending=False).head(5).reset_index()
            similar_movies  = similar_movies.sort_values(by='year',ascending=False).reset_index()
            candidate_movies = similar_movies.values.tolist()
            movie_titles_list = [lst[2] for i, lst in enumerate(candidate_movies)]
            movie_titles_list = self.diff(movie_titles_list,recommended_movies)
            #movie_titles_list = self.unique(movie_titles_list)
            movie_titles_list = movie_titles_list[:nb_recom]

        except:
            movie_titles_list =[]

        print(movie_titles_list)
        return movie_titles_list

    def diff(self, li1, li2):
        li_dif = [i for i in li1 if i.split('(')[0] not in li2]
        return li_dif

    def unique(self, list1):
        # insert the list to the set
        list_set = set(list1)
        # convert the set to the list
        unique_list = (list(list_set))
        return unique_list

if __name__ == '__main__':
    obj= Recommender()
    rec= ['abc']
    recommendations = obj.get_similar_movies_based_on_content(141566,3,rec)
    print(recommendations)
