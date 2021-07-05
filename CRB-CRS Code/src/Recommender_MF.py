import pandas as pd
import numpy as np
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import linear_kernel
import random
from numpy import asarray
from numpy import save
from numpy import load
import os

class Recommender_MF():



    def __init__(self):
        self.PATH = os.path.dirname(os.path.abspath(__file__))
        self.ROOT_DIR_PATH = os.path.abspath(os.path.dirname(self.PATH))
        self.ITEM_DATA_path = os.path.join(self.ROOT_DIR_PATH, 'data\\recommenders_item_data\\')
        self.model =np.array([])
        self.loaded_model =np.array([])
        self.movie_title_list = None
        self.df_ratings = pd.DataFrame()
        self.df_movies = pd.DataFrame()
        self.movies_mentions = pd.DataFrame()
        self.is_session_changed = False
        self.recommended_movies = []
        #self.df_ratings = pd.read_csv(self.ITEM_DATA_path+ "ratings_latest.csv")
        self.df_movies = pd.read_csv(self.ITEM_DATA_path+"dfmovies.csv")
        self.movies_mentions = pd.read_csv(self.ITEM_DATA_path+"movies_data.csv", encoding="utf-8")
        with open(self.ITEM_DATA_path+'movie_titles.txt', 'r', encoding='utf-8') as filehandle:
            self.movie_title_list = filehandle.readlines()
        self.movie_title_list = [i.strip() for i in self.movie_title_list]
        self.load_model()
        if self.loaded_model.size == 0:
            self.df_ratings = pd.read_csv(self.ITEM_DATA_path+ "ratings_latest.csv", usecols=['userId', 'movieId', 'rating'])
            self.data_initialization()
            self.load_model()

    def store_model(self, data):
        # save numpy array as npy file
        # save to npy file
        save(self.ITEM_DATA_path+'model.npy', data)
        print('model data saved')

    def load_model(self):
        # load numpy array from npy file
        self.loaded_model = load(self.ITEM_DATA_path+'model.npy')
        print('model data loaded')

    def data_initialization(self):
        self.df_ratings = self.df_ratings[:15000000]
        combine_movie_rating = pd.merge(self.df_ratings, self.df_movies, on='movieId')
        combine_movie_rating = combine_movie_rating.dropna(axis = 0, subset = ['title'])
        movie_ratingCount = (combine_movie_rating.
             groupby(by = ['title'])['rating'].
             mean().
             reset_index().
             rename(columns = {'rating': 'ratingMean'})
             [['title', 'ratingMean']]
            )
        rating_with_totalRatingCount = combine_movie_rating.merge(movie_ratingCount, left_on = 'title', right_on = 'title', how = 'left')
        user_rating = rating_with_totalRatingCount.drop_duplicates(['userId','title'])

        df_temp_rating_count = user_rating.drop_duplicates(['title'],keep='first')
        year_list = []
        for index, row in self.df_movies.iterrows():
            try:
                title = str(row['title'])
                if title.__contains__('(') and title.__contains__(')'):
                    year = int(title[len(title)-5:].replace(')',''))
                    year_list.append(year)
                else:
                    year = 0000
                    year_list.append(year)
            except:
                year_list.append(0000)
                continue

        self.df_movies['year'] = year_list
        self.df_movies= pd.merge(self.df_movies[['movieId','title','genres','year']], df_temp_rating_count[['title','ratingMean']], left_on = 'title', right_on = 'title', how = 'left').fillna(0)
        self.df_movies.to_csv(self.ITEM_DATA_path+'dfmovies.csv')
        movie_user_rating_pivot = pd.pivot_table(user_rating, index = 'userId', columns = 'title', values = 'rating').fillna(0)
        X = movie_user_rating_pivot.values.T
        SVD = TruncatedSVD(n_components=20, random_state=10)
        matrix = SVD.fit_transform(X)
        self.model = np.corrcoef(matrix)
        movie_title = movie_user_rating_pivot.columns
        movie_titles_list = list(movie_title)
        with open(self.ITEM_DATA_path+'movie_titles_o.txt', 'w') as file:
            file.writelines(["%s\n" % item  for item in movie_titles_list])
        self.store_model(self.model)
        print('model data saved!')


    def get_similar_movies_based_on_user_ratings(self, input_movieId, nb_rec, recommended_movies):
        try:

            movies_list =[]
            title = self.movies_mentions.loc[self.movies_mentions['databaseId'] == int(input_movieId)]['title']
            title = title.iloc[0]
            if len(title) < 2:
                return movies_list
            input_genres = self.movies_mentions.loc[self.movies_mentions['databaseId'] == int(input_movieId)]['genres'].iloc[0].split('|')
            movie_index = self.movie_title_list.index(title)
            sim_scores  = list(enumerate(self.loaded_model[movie_index]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:15]
            movie_indices = [i[0] for i in sim_scores]
            movie_sim_scores = [i[1] for i in sim_scores]
            similar_movies = pd.DataFrame([self.movie_title_list[i] for i in movie_indices],columns=['title'])
            similar_movies = pd.merge(similar_movies[['title']],self.df_movies[['title','genres','year','ratingMean']],how='left',on='title')
            similar_movies =  self.match_genres(input_genres, similar_movies)

            similar_movies = similar_movies.sort_values(by ='ratingMean', ascending=False).head(10).reset_index()
            similar_movies  = similar_movies.sort_values(by ='year', ascending=False).reset_index()
            similar_movies  = similar_movies.sort_values(by ='matchCount', ascending=False).head(5)
            similar_movies = similar_movies[similar_movies.year != 1500]
            candidate_movies = similar_movies.values.tolist()
            movie_titles_list = [lst[2] for i, lst in enumerate(candidate_movies)]
            movie_titles_list = self.diff(movie_titles_list, recommended_movies)
            movie_titles_list = movie_titles_list[:nb_rec]

        except:
            movie_titles_list= []
        print(movie_titles_list)
        return movie_titles_list
    def match_genres(self, input_genre_list, recommended_list):
        matching_count_list = []
        for row in recommended_list.iterrows():
            rec_genres_list = row[1]['genres'].split('|')
            count=  len(list(set(input_genre_list).intersection(rec_genres_list)))
            matching_count_list.append(count)

        recommended_list['matchCount'] = matching_count_list
        return recommended_list


    def unique(self, list1):
        # insert the list to the set
        list_set = set(list1)
        # convert the set to the list
        unique_list = (list(list_set))
        return unique_list

    def filter_duplicate_movies(self,rec_list, mentioned_list):
        unique_movies = list(set(rec_list).difference(mentioned_list))
        return unique_movies

    def diff(self, li1, li2):
        li_dif = [i for i in li1 if i.split('(')[0] not in li2]
        return li_dif

if __name__ == '__main__':
    obj= Recommender_MF()
    rec=['abc']
    recommendations = obj.get_similar_movies_based_on_user_ratings(131178, 1, rec)
    recommendations = obj.get_similar_movies_based_on_user_ratings(184098, 2, recommendations)
    print(recommendations)

