import pandas as pd
import numpy as np
from surprise import Dataset, Reader, KNNBasic, KNNWithMeans, KNNBaseline, NMF, SVD, SVDpp, CoClustering, BaselineOnly
import matplotlib.pyplot as plt

# Add a guassian noise to a dataframe
def add_gaussian_noise(df, noise):
    noisy_df = df.copy()
    noisy_df['rating'] += np.random.normal(0, noise, size=len(noisy_df))
    noisy_df['rating'] = np.clip(noisy_df['rating'], 1, 5)  # Keep the ratings between 1 and 5
    noisy_df['rating'] = noisy_df['rating'].round().astype(int)
    return noisy_df


# Returns a complete dataframe of predicted ratings based on a trained model and an incomplete dataframe
def predict_all_ratings(model, df):
    user_ids = df['user id'].unique()
    movie_ids = df['movie id'].unique()
    predictions = []

    for user_id in user_ids:
        for movie_id in movie_ids:
            prediction = model.predict(user_id, movie_id).est
            predictions.append({'user id': user_id, 'movie id': movie_id, 'rating': prediction})

    predictions_df = pd.DataFrame(predictions)
    return predictions_df



# Trains model on a dataframe
def train_model(model_name,df):
    models_dict = {
        "KNN": KNNBasic(verbose=False),
        "KNN with Means": KNNWithMeans(verbose=False),
        "KNN Baseline": KNNBaseline(verbose=False),
        "NMF": NMF(verbose=False),
        "SVD (ALS)": SVD(verbose=False),
        "SVD++": SVDpp(verbose=False),
        "Co-clustering": CoClustering(verbose=False),
        "BaselineOnly": BaselineOnly(verbose=False)
    }
    # Check if the model exists in the dictionary
    if model_name in models_dict:
      # Get the model associated to the it's name
      model = models_dict[model_name]
      # Transform the data for the surprise library
      reader = Reader(rating_scale=(1, 5))
      data = Dataset.load_from_df(df[['user id', 'movie id', 'rating']], reader)
      trainset = data.build_full_trainset() # We don't need a test set, we train on the whole dataset
      # Fit the model on the train set
      model.fit(trainset)
      return model
    # Return None if the model name doesn't exist in the dictionary
    else:
      print("Unknown model.")
      return None



# Returns the rmse between two dataframes
def calculate_rmse(df1, df2):
    # Merge df1 and df2 on 'user id' and 'movie id' to get ratings from df2 for corresponding entries in df1
    merged_df = df1.merge(df2, on=['user id', 'movie id'], suffixes=('_df1', '_df2'), how='left')

    # Calculate squared errors only for existing ratings in df2
    squared_errors = (merged_df['rating_df1'] - merged_df['rating_df2']) ** 2

    # Calculate RMSE
    mse = squared_errors.mean()
    rmse = np.sqrt(mse)

    return rmse





# Convert the Oracle to movielens format (user id, movie id, rating)
def convert_to_movielens_format(oracle):
    # Convert the ratings numpy matrix into pandas dataframe
    num_films = oracle.shape[1]
    ratings_df = np.rint(pd.DataFrame(oracle, columns=[f"Film {i+1}" for i in range(num_films)]))
    ratings_df.index.name = 'User'
    # Initialize lists to store user IDs, movie IDs, and ratings
    user_ids = []  # List to store user IDs
    movie_ids = []  # List to store movie IDs
    ratings = []  # List to store ratings
    num_users, num_movies = ratings_df.shape
    # Loop through each user and movie in the ratings dataframe
    for user in range(num_users):
        for movie in range(num_movies):
            # Get user IDs from the oracle, starting from 0
            user_ids.append(user)
            # Get the movie IDs from the oracle , starting from 1
            movie_ids.append(movie + 1)
            # Get the rating from the Oracle
            rating = ratings_df.iloc[user, movie]
            ratings.append(rating)
    # Create a DataFrame in MovieLens format
    movielens_df = pd.DataFrame({
        'user id': user_ids,
        'movie id': movie_ids,
        'rating': ratings
    })
    # Convert the rounded ratings to integers
    movielens_df['rating'] = movielens_df['rating'].round().astype(int)
    return movielens_df



# Returns the rmse of each model on a specific configuration of a complete dataset
def evaluate_model_on_complete_oracle(model, complete_oracle_df, density_percentage, noise):
  density = int(len(complete_oracle_df) * density_percentage) # Get the exact number of ratings to take based on the percentage
  trainset = complete_oracle_df.sample(n=density, random_state=42)
  testset = complete_oracle_df.drop(trainset.index)
  noisy_trainset = add_gaussian_noise(trainset, noise)
  noisy_trainset['rating'] = noisy_trainset['rating'].round().astype(int) # TO VERIFY
  model = train_model(model,noisy_trainset)
  predicted_df = predict_all_ratings(model, testset)
  rmse = calculate_rmse(predicted_df, complete_oracle_df)
  return rmse

# Returns the rmse of each model on a specific configuration of an incomplete dataset
def evaluate_model_on_uncomplete_oracle(model, df, density_percentage, noise):
    density = int(len(df) * density_percentage)
    trainset = df.sample(n=density, random_state=42)
    testset = df.drop(trainset.index)
    noisy_trainset = add_gaussian_noise(trainset, noise)
    noisy_trainset['rating'] = noisy_trainset['rating'].round().astype(int) # TO VERIFY
    model = train_model(model,noisy_trainset)
    predicted_df = predict_all_ratings(model, testset)
    rmse = calculate_rmse(predicted_df, testset)
    return rmse



def get_model_performance(model, oracle, density_percentage, noise, complete_oracle):
  if complete_oracle == 0:
    rmse = evaluate_model_on_uncomplete_oracle(model, oracle, density_percentage, noise)
    return rmse

  else:
    rmse = evaluate_model_on_complete_oracle(model, oracle, density_percentage, noise)
    return rmse


def plot_results(df_results, models, densities, noise_levels):
    # Plot RMSE based on the noise level
    plt.figure(figsize=(10, 6))
    for model_name in models:
        for density_percentage in densities:
            df_filtered = df_results[(df_results['model'] == model_name) & (df_results['num_ratings'] == int(len(df_results) * density_percentage))]
            plt.plot(df_filtered['noise_level'], df_filtered['rmse'], marker='o', label=f"{model_name}, {int(len(df_results) * density_percentage)} ratings")

    plt.xlabel('Noise Level')
    plt.ylabel('RMSE')
    plt.title('RMSE based on the Noise Level')
    plt.legend()
    plt.grid(True)
    plt.xticks(noise_levels)
    plt.show()

    # Plot RMSE based on the number of ratings
    plt.figure(figsize=(10, 6))
    for model_name in models:
        for noise_level in noise_levels:
            df_filtered = df_results[(df_results['model'] == model_name) & (df_results['noise_level'] == noise_level)]
            plt.plot(df_filtered['num_ratings'], df_filtered['rmse'], marker='o', label=f"{model_name}, Noise Level {noise_level}")

    plt.xlabel('Training Ratings')
    plt.ylabel('RMSE')
    plt.title('RMSE based on the Number of Ratings')
    plt.legend()
    plt.grid(True)
    plt.xticks(df_results['num_ratings'], labels=[f"{density*100}% ({num_ratings})" for density, num_ratings in zip(densities, df_results['num_ratings'])])
    plt.show()