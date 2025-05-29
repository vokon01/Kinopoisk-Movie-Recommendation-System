from flask import Flask, render_template, url_for, request
import ast
import os
import pandas as pd
import numpy as np
from surprise import SVD, dump
import random
import pickle
import ast
import pathlib

def get_cbs_recommendations(idx, num_movies = 10):
    idx = indices[idx]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_movies]
    movie_indices = [i[0] for i in sim_scores]
    return ids.iloc[movie_indices]

proj_folder = pathlib.Path('C:/Users/vokon/PycharmProjects/PythonProject3')
rec_movies = pd.read_csv(proj_folder.joinpath('datasets/processed', 'rec_movies.csv'))
ratings = pd.read_csv(proj_folder.joinpath('datasets/processed', 'rating_df.csv'))

with open(proj_folder.joinpath('models', 'ids.pkl'), 'rb') as f:
  ids = pickle.load(f)

with open(proj_folder.joinpath('models', 'indices.pkl'), 'rb') as f:
  indices = pickle.load(f)

with open(proj_folder.joinpath('models', 'cosine_sim.npy'), 'rb') as f:
  cosine_sim = np.load(f)

_, cf_mod_model = dump.load(proj_folder.joinpath('models', 'cfs_1_model'))

_, cf_mem_model = dump.load(proj_folder.joinpath('models', 'cfs_2_model'))

popular_picks = rec_movies.sort_values(by = ['global_rank'], ascending=True).head(100).sample(n=8)
pop_movies = ast.literal_eval(popular_picks.loc[:, ['name', 'description', 'poster']].head(8).to_json(force_ascii=False, orient="records"))
pop_data = []
for movie in pop_movies:
    pop_data.append([movie['name'], movie['description'][0:100],
                     os.path.normpath(ast.literal_eval(movie['poster'])['previewUrl']).replace('\\', '/'),
                     'https://www.youtube.com/results?search_query=' + movie['name'].replace(' ', '+') + '+трейлер'])
# print(cbs_movies)

users_with_recomendations = pd.Series(ratings['userId'].unique().astype('int'))
# current_user = random.choice(users_with_recomendations)
# print(current_user)

def get_recommendation(current_user, rec_movies):
    cbs_rec_movies = ratings.loc[(ratings['userId'] == current_user) & (ratings['rating'] > 3)].sort_values('timestamp', ascending = False).iloc[:10]['movieId'].apply(get_cbs_recommendations)
    cbs_rec_movies = cbs_rec_movies.stack().reset_index().drop(columns = ['level_1']).set_index('level_0')
    # Generating list of already watched movies
    watched_movies = ratings.loc[(ratings['userId'] == current_user)].sort_values('timestamp', ascending = False)['movieId']
    rec_movies = rec_movies.loc[(~rec_movies['id'].isin(watched_movies))]
    # rec_movies = rec_movies[['id', 'name', 'description', 'genres']]
    # rec_movies = pd.merge(rec_movies, movies_with_rank, how = 'left', on=['id']).drop_duplicates()
    rec_movies.loc[:,'cbs_rec'] = rec_movies['id'].isin(cbs_rec_movies[0])
    rec_movies.loc[:, 'cfs_mod_est'] =rec_movies['id'].apply(lambda x: cf_mod_model.predict(current_user,x).est)
    rec_movies.loc[:, 'cfs_mod_est'] = rec_movies.apply(lambda x: x['cfs_mod_est'] if (x.cfs_est_flag) else 0, axis=1)
    rec_movies.loc[:, 'cfs_mem_est'] = rec_movies['id'].apply(lambda x: cf_mem_model.predict(current_user,x).est)
    # rec_movies.loc[:, 'cf_mem_details'] =rec_movies['id'].apply(lambda x: cf_mem_model.predict(current_user,x).details).apply(details_cleanup)
    rec_movies.loc[:, 'cf_mem_est'] =rec_movies.apply(lambda x: x['cfs_mem_est'] if (x.cfs_est_flag) else 0, axis=1)
    rec_movies.loc[:, 'total_score'] = rec_movies['cfs_mod_est'] + rec_movies['cfs_mem_est']
    # Generating top picks
    cfs_picks = rec_movies.sort_values(by = ['total_score', 'global_rank', 'cbs_rec'], ascending=[False, True, False]).head(8)
    cbs_picks = rec_movies.sort_values(by = ['cbs_rec', 'global_rank'], ascending=[False, True]).head(8)
    cfs_movies = ast.literal_eval(cfs_picks.loc[:, ['name', 'description', 'poster']].head(8).to_json(force_ascii=False, orient="records"))
    cfs_data = []
    for movie in cfs_movies:
        cfs_data.append([movie['name'],movie['description'][0:100],os.path.normpath(ast.literal_eval(movie['poster'])['previewUrl']).replace('\\', '/'), 'https://www.youtube.com/results?search_query='+movie['name'].replace(' ', '+')+'+трейлер'])
        # print((ast.literal_eval(movie['poster'])['url']).replace('/', ''))
    # print(data)
    cbs_movies = ast.literal_eval(cbs_picks.loc[:, ['name', 'description', 'poster']].head(8).to_json(force_ascii=False, orient="records"))
    cbs_data = []
    for movie in cbs_movies:
        cbs_data.append([movie['name'],movie['description'][0:100],os.path.normpath(ast.literal_eval(movie['poster'])['previewUrl']).replace('\\', '/'), 'https://www.youtube.com/results?search_query='+movie['name'].replace(' ', '+')+'+трейлер'])
    # print(cbs_movies)
    return cfs_data, cbs_data


app = Flask(__name__)

@app.route("/")
def main():
    # movies = ast.literal_eval(rec_movies.loc[:, ['name', 'description', 'poster']].head(5).to_json(force_ascii=False, orient="records"))
    # return render_template("index_2.html", cfs_data =cfs_data, cbs_data =cbs_data)
    return  render_template("home.html", pop_data=pop_data)

@app.route("/recom_by_userid", methods=['post'])
def recom_by_userid():
    current_user = random.choice(users_with_recomendations)
    cfs_data, cbs_data = get_recommendation(current_user, rec_movies)
    # return  str(current_user)
    return render_template("index_2.html", cfs_data =cfs_data, cbs_data =cbs_data)


if __name__ == "__main__":
    app.run(debug=True)