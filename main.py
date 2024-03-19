import pandas as pd
from functions import *
from preferences_mean import *
from matrix_factorization import *
from random_trees import *

# Oracle settings (1,586,126 rating)
num_users = 943
num_movies = 1682


# Method 1 : Generate a complete oracle with matrix factorization
#oracle = generate_factorized_matrix(num_users, num_movies)
#movielens_format_oracle = convert_to_movielens_format(oracle) # rinted values
#complete_oracle = 1  # 1 if we have got a complete matrix of ratings and 0 if not.

# Method 2 : Generate a complete oracle with preferences means
#genres = ["Action", "Comédie", "Drame", "Science-Fiction", "Horreur"]
#oracle = generate_oracle(num_users, num_movies, genres)
#movielens_format_oracle = convert_to_movielens_format(oracle) # rinted values
#complete_oracle = 1  # 1 if we have got a complete matrix of ratings and 0 if not.

# Method 3 : Generate a complete oracle with random trees
genres = ["Action", "Comédie", "Drame", "Science-Fiction", "Horreur"]
leaf_probability = 0.3
oracle = generate_ratings_matrix(num_users, num_movies, genres, leaf_probability)
movielens_format_oracle = convert_to_movielens_format(oracle) # rinted values
complete_oracle = 1  # 1 if we have got a complete matrix of ratings and 0 if not.

# Method 4 : import an existing dataset with missing ratings
#df = pd.read_csv("movielens100k.csv")
#complete_oracle = 0  # 1 if we have got a complete matrix of ratings and 0 if not.


# Set the final dataframe that we will use
df = movielens_format_oracle


# Set the testing parameters
densities = [0.9,0.8,0.7]
noise_levels = [0.0,0.5,1.0]
models = ["SVD++","SVD (ALS)", "KNN"]

# Store the results
results = []

# Loop on the models
for model_name in models:
  print(f"Model : {model_name}")
  for density_percentage in densities:
    print (f"percentage of training ratings : {density_percentage*100}%")
    for noise_level in noise_levels:
      print(f"Noise Level : {noise_level}")
      rmse = get_model_performance(model_name, df, density_percentage, noise_level, complete_oracle)
      num_training_ratings = int(len(df) * density_percentage)
      print (f"RMSE of {model_name} on the oracle of noise {noise_level} , and {density_percentage*100}% = ({num_training_ratings}) ratings is : {rmse:.4f}")
      # Store the results
      results.append({
        'model': model_name,
        'num_ratings': num_training_ratings,
        'noise_level': noise_level,
        'rmse': rmse
        })


plot_results(results,noise_levels,densities,models)