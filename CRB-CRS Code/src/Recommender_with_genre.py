import pandas as pd
import numpy as np
from ast import literal_eval
import random
import os

class Recommender_with_genre:

    def __init__(self):
            self.PATH = os.path.dirname(os.path.abspath(__file__))
            self.ROOT_DIR_PATH = os.path.abspath(os.path.dirname(self.PATH))
            self.ITEM_DATA_path = os.path.join(self.ROOT_DIR_PATH, 'data\\recommenders_item_data\\')
            self.df_movie_content = pd.DataFrame()
            self.recommended_movies=[]
            self.C = 0
            self.m =0
            if self.df_movie_content.__len__() == 0:
                self.df_movie_content = self.data_initialization()

    def get_similar_movies_based_on_genre(self, genre, nb_recom, recommended_movies, percentile=0.85):
        try:
            movie_titles_list = []
            print('the genre is ' + genre)
            if genre.lower() == 'scary':
                genre ='Horror'
            elif genre.lower() =='romantic' or genre.lower()=='romances':
                genre='Romance'
            elif genre.lower() == 'preference':
                genre = 'Adventure'
            elif genre.lower() =='suspense':
                genre = 'Thriller'
            elif genre.lower() =='funny':
                genre = 'Comedy'
            elif genre.lower() == 'comedies':
                genre = 'Comedy'
            elif genre.lower() == 'scifi':
                genre = 'Science Fiction'
            elif genre.lower() == 'kids':
                genre = 'Comedy'
            elif genre.lower() == 'mysteries':
                genre = 'mystery'

            genre =genre.title()
            df = self.df_movie_content[self.df_movie_content['genre'] == genre]
            vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
            vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
            self.C = vote_averages.mean()
            self.m = vote_counts.quantile(.85)

            col_list = ['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genre']
            qualified = df[(df['vote_count'] >= self.m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][col_list]
            qualified['vote_count'] = qualified['vote_count'].astype('int')
            qualified['vote_average'] = qualified['vote_average'].astype('int')

            qualified['weighted_rating'] = qualified.apply(lambda x:
                                                           (x['vote_count']/(x['vote_count']+self.m) * x['vote_average']) + (self.m/(self.m+x['vote_count']) * self.C),
                                                           axis=1)
            qualified = qualified.sort_values("weighted_rating", ascending=False).head(10).reset_index()
            qualified = qualified.sort_values('year', ascending=False)
            movies_list = qualified.values.tolist()
            movie_titles_list = [lst[1]+ ' ('+lst[2]+')' for i, lst in enumerate(movies_list)]
            movie_titles_list = self.diff(movie_titles_list, recommended_movies)
            #movie_titles_list = self.unique(movie_titles_list)
            movie_titles_list = movie_titles_list[:nb_recom]

        except (RuntimeError, TypeError, NameError) as err:
            print(err)
            print("exception accured here")
            movie_titles_list =[]
        print(movie_titles_list)
        return movie_titles_list

    def unique(self, list1):
        # insert the list to the set
        list_set = set(list1)
        # convert the set to the list
        unique_list = (list(list_set))
        return unique_list
    def weighted_rating(x):
        v = x['vote_count']
        R = x['vote_average']
        return (v/(v+self.m) * R) + (self.m/(self.m+v) * self.C)

    def diff(self, li1, li2):
        li_dif = [i for i in li1 if i.split('(')[0] not in li2]
        return li_dif

    def data_initialization(self):
        m_df = pd.read_csv(self.ITEM_DATA_path + 'movies_metadata.csv', low_memory=False)
        m_df['genres'] = m_df['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x]
                                                                               if isinstance(x, list) else [])
        # extracting release year from release_date
        m_df['year'] = pd.to_datetime(m_df['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0]
                                                                                   if x != np.nan else np.nan)

        #Now, building list for particular genres. For that, cutoff is relaxed to 85% instead of 95
        temp = m_df.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
        temp.name = 'genre'
        mges_df = m_df.drop('genres', axis=1).join(temp)
        return mges_df

if __name__ == '__main__':
    obj = Recommender_with_genre()
    rec= ['abcv']
    rec_list  = obj.get_similar_movies_based_on_genre('Mystery', 2, rec)
    print(rec_list)



