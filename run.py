from os import path
from elliot.run import run_experiment
from data_preprocessing import movielens_preprocessing, facebook_book_preprocessing, yahoo_movies_preprocessing
from config_fb_template import TEMPLATE as fb_template
from config_movielens_template import TEMPLATE as movielens_template
from config_yahoo_template import TEMPLATE as yahoo_template
import os

movielens_data_folder = './data/movielens'
facebook_book_folder = './data/facebook_book'
yahoo_movies_folder = './data/yahoo_movies'

CONFIG_DIR = './config_files'
assert os.path.exists(CONFIG_DIR)

# PRE-PROCESSING
facebook_book_preprocessing.run(data_folder=facebook_book_folder)
yahoo_movies_preprocessing.run(data_folder=yahoo_movies_folder)
movielens_preprocessing.run(data_folder=movielens_data_folder)


# RUN EXPERIMENTS
alphas = [str(i) for i in [0, 0.2, 0.4, 0.6, 0.8, 1]]

# - run the experiments for Facebook Books
dataset = 'facebook_book'
for a in alphas:
    config = fb_template.format(dataset, dataset, dataset, dataset, alpha=a)
    name = dataset + '_iron_tore' + '_' + a + '.yml'
    config_path = os.path.join(CONFIG_DIR, name)
    with open(config_path, 'w') as file:
        file.write(config)
    run_experiment(config_path)


# - run the experiments for Yahoo Movies
dataset = 'yahoo_movies'
for a in alphas:
    config = yahoo_template.format(dataset, dataset, dataset, dataset, alpha=a)
    name = dataset + '_iron_tore' + '_' + a + '.yml'
    config_path = os.path.join(CONFIG_DIR, name)
    with open(config_path, 'w') as file:
        file.write(config)
    run_experiment(config_path)


# - run the experiments for MovieLens 1M
dataset = 'movielens'
for a in alphas:
    config = movielens_template.format(dataset, dataset, dataset, dataset, alpha=a)
    name = dataset + '_iron_tore' + '_' + a + '.yml'
    config_path = os.path.join(CONFIG_DIR, name)
    with open(config_path, 'w') as file:
        file.write(config)
    run_experiment(config_path)










