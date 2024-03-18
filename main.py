import pandas as pd
import matplotlib.pyplot as plt
from functions import *
from preferences_mean import *
from matrix_factorization import *
from random_trees import *

# Oracle settings (1,586,126 rating)
num_users = 943
num_movies = 1682

# List of the possible genres
genres = ["Action", "Comédie", "Drame", "Science-Fiction", "Horreur"]

# Probability to get a leaf
leaf_probability = 0.3

# Method 1 : Generate a complete oracle with matrix factorization
#oracle = generate_factorized_matrix(num_users, num_movies)
#movielens_format_oracle = convert_to_movielens_format(oracle) # rinted values

# Method 2 : Generate a complete oracle with preferences means
#oracle = generate_oracle(num_users, num_movies, genres)
#movielens_format_oracle = convert_to_movielens_format(oracle) # rinted values

# Method 3 : Generate a complete oracle with random trees
oracle = generate_ratings_matrix(num_users, num_movies, genres, leaf_probability)
movielens_format_oracle = convert_to_movielens_format(oracle) # rinted values

# Method 4 : import an existing dataset with missing ratings
#df = pd.read_csv("movielens100k.csv")


# Set the final dataframe that we will use
df = movielens_format_oracle


# Set the testing parameters
densities = [0.9,0.8,0.7]
noise_levels = [0.0,0.5,1.0]
models = ["SVD++","SVD (ALS)", "KNN"]

complete_oracle = 1  # 1 if we have got a complete matrix of ratings and 0 if not.

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


# Convert results into a pandas dataframe
df_results = pd.DataFrame(results)

# First graph: RMSE based on the noise
plt.figure(figsize=(10, 6))
for model_name in models:
    for density_percentage in densities:
        df_filtered = df_results[(df_results['model'] == model_name) & (df_results['num_ratings'] == int(len(df) * density_percentage))]
        plt.plot(df_filtered['noise_level'], df_filtered['rmse'], marker='o', label=f"{model_name}, {int(len(df) * density_percentage)} ratings")

plt.xlabel('Noise Level')
plt.ylabel('RMSE')
plt.title('RMSE based on the Noise Level')
plt.legend()
plt.grid(True)

# values to print on the X axis
plt.xticks(noise_levels)  # Use noise_levels directly as ticks


# Second graph: RMSE based on the number of ratings
plt.figure(figsize=(10, 6))
for model_name in models:
    for noise_level in noise_levels:
        df_filtered = df_results[(df_results['model'] == model_name) & (df_results['noise_level'] == noise_level)]
        plt.plot(df_filtered['num_ratings'], df_filtered['rmse'], marker='o', label=f"{model_name}, Noise Level {noise_level}")

plt.xlabel('Number of Ratings')
plt.ylabel('RMSE')
plt.title('RMSE based on the Number of Ratings')
plt.legend()
plt.grid(True)

# values to print on the X axis
x_ticks_labels = [f"{density_percentage*100}% ({int(len(df) * density_percentage)})" for density_percentage in densities]
plt.xticks([int(len(df) * density_percentage) for density_percentage in densities], x_ticks_labels)

plt.show()