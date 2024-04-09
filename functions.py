import pandas as pd
import numpy as np
from surprise import Dataset, Reader, KNNBasic, KNNWithMeans, KNNBaseline, NMF, SVD, SVDpp, CoClustering, BaselineOnly, accuracy
import matplotlib.pyplot as plt
import copy


# Add a guassian noise to a dataframe
def add_gaussian_noise(df, noise):
    noisy_df = df.copy()
    noisy_df['rating'] += np.random.normal(0, noise, size=len(noisy_df))
    noisy_df['rating'] = np.clip(noisy_df['rating'], 1, 5)  # Keep the ratings between 1 and 5
    noisy_df['rating'] = noisy_df['rating'].round().astype(int)
    return noisy_df


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




# Returns the rmse of each model on a specific configuration of a complete dataset
def evaluate_model_on_complete_oracle(model, df, training_percentage, noise):
    num_training_ratings = int(
    len(df) * training_percentage)  # Get the exact number of ratings to take based on the percentage
    trainset = df.sample(n=num_training_ratings, random_state=42)
    testset = df.drop(trainset.index)
    noisy_trainset = add_gaussian_noise(trainset, noise)
    noisy_trainset['rating'] = noisy_trainset['rating'].round().astype(int)
    num_models_to_train = 5
    training_percentage = 0.9
    epsilon = 0.05
    models_list = combined_bagging_train(model, noisy_trainset, num_models_to_train, training_percentage, epsilon)
    predicted_df = predict_all_ratings_combined_baggings(models_list, testset)
    rmse = calculate_rmse(predicted_df, df)
    return rmse

# Returns the rmse of each model on a specific configuration of an incomplete dataset
def evaluate_model_on_uncomplete_oracle(model, df, training_percentage, noise):
    num_training_ratings = int(len(df) * training_percentage)
    trainset = df.sample(n=num_training_ratings, random_state=42)
    testset = df.drop(trainset.index)
    noisy_trainset = add_gaussian_noise(trainset, noise)
    noisy_trainset['rating'] = noisy_trainset['rating'].round().astype(int)
    model = train_model(model,noisy_trainset)
    predicted_df = predict_all_ratings(model, testset)
    rmse = calculate_rmse(predicted_df, testset)
    return rmse



def get_model_performance(model, oracle, training_percentage, noise, complete_oracle):
  if complete_oracle == 0:
    rmse = evaluate_model_on_uncomplete_oracle(model, oracle, training_percentage, noise)
    return rmse

  else:
    rmse = evaluate_model_on_complete_oracle(model, oracle, training_percentage, noise)
    return rmse



# ---------------------- GRAPHS DISPLAY --------------------------- #



def plot_results(results,noise_levels,densities,models,df) :

  # Convert results into a pandas dataframe
  df_results = pd.DataFrame(results)

  # First graph: RMSE based on the noise
  plt.figure(figsize=(10, 6))
  for model_name in models:
      for training_percentage in densities:
          df_filtered = df_results[(df_results['model'] == model_name) & (df_results['num_ratings'] == int(len(df) * training_percentage))]
          plt.plot(df_filtered['noise_level'], df_filtered['rmse'], marker='o', label=f"{model_name}, {int(len(df) * training_percentage)} ratings")

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
  x_ticks_labels = [f"{training_percentage*100}% ({int(len(df) * training_percentage)})" for training_percentage in densities]
  plt.xticks([int(len(df) * training_percentage) for training_percentage in densities], x_ticks_labels)

  plt.show()




  # ----------------- BAGGING TRAINING ------------------- #




def combined_bagging_train(model_name, df, num_models, training_percentage, epsilon):
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

    num_training_ratings = int(len(df) * training_percentage)
    if model_name in models_dict:
        base_model = models_dict[model_name]
        models = []
        weights = []  # List to store weights for each instance

        for _ in range(num_models):
            # Train initial model
            model = copy.deepcopy(base_model)
            sample_df = df.sample(n=num_training_ratings)
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(sample_df[['user id', 'movie id', 'rating']], reader)
            trainset = data.build_full_trainset()
            model.fit(trainset)
            models.append(model)

            # Weighted resampling for subsequent models
            if len(models) > 1: # We need at least 2 models to do the bagging
                updated_weights = []  # Updated weights for the next training sample
                for _, row in df.iterrows():
                    user_id = row['user id']
                    movie_id = row['movie id']
                    rating = row['rating']

                    # Predict using all models except the last one
                    median_prediction = combined_bagging_predict(models,user_id,movie_id)

                    # Calculate error and update weights
                    error = abs(median_prediction - rating)
                    updated_weight = error if error <= epsilon else error + epsilon
                    updated_weights.append(updated_weight)

                # Normalize updated weights
                updated_weights_sum = sum(updated_weights)
                updated_weights = [w / updated_weights_sum for w in updated_weights]
                weights = updated_weights

                # Resample using updated weights
                indices = np.random.choice(df.index, size=num_training_ratings, replace=True, p=weights)
                df = df.loc[indices].reset_index(drop=True)

        return models
    else:
        print("Modèle inconnu.")
        return None


# Define bagging_predict to use median of all model predictions
def combined_bagging_predict(models, user_id, movie_id):
    predictions = [model.predict(user_id, movie_id).est for model in models]
    median_prediction = np.median(predictions)
    return median_prediction



# Returns a complete dataframe of predicted ratings based on a trained model and an incomplete dataframe
def predict_all_ratings_combined_baggings(models_list, df):
    user_ids = df['user id'].unique()
    movie_ids = df['movie id'].unique()
    predictions = []

    for user_id in user_ids:
        for movie_id in movie_ids:
            prediction = combined_bagging_predict(models_list,user_id, movie_id)
            predictions.append({'user id': user_id, 'movie id': movie_id, 'rating': prediction})

    predictions_df = pd.DataFrame(predictions)
    return predictions_df
