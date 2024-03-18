import pandas as pd
import numpy as np
from surprise import Dataset, Reader, KNNBasic, KNNWithMeans, KNNBaseline, NMF, SVD, SVDpp, CoClustering, BaselineOnly, accuracy
import matplotlib.pyplot as plt
import random


# --------------------------------------- DOUDA ---------------------------------- #

# Fonction pour créer un arbre aléatoire avec une probabilité spécifiée pour les feuilles
def creer_arbre(genres_disponibles, probabilite_feuille=0.5):
    if not genres_disponibles or random.random() < probabilite_feuille:
        # Si la liste de genres est vide ou si la probabilité est atteinte, retourne une note aléatoire
        return random.randint(1, 5)

    # Choix aléatoire d'un genre parmi les genres disponibles
    genre = random.choice(genres_disponibles)
    genres_disponibles.remove(genre)

    # Création d'un score aléatoire pour le genre
    score_genre = round(random.uniform(0, 1), 1)  # Score aléatoire entre 0 et 1 avec un chiffre après la virgule

    # Création du nœud avec le genre choisi et son score
    arbre = {"genre": genre, "score": score_genre, "branches": {}}

    # Création de deux branches pour chaque nœud
    for i in range(2):
        # Choix aléatoire entre une note ou un nouveau nœud
        if random.random() < probabilite_feuille:
            # Création d'une note aléatoire entre 1 et 5
            arbre["branches"][f"Branche {i}"] = random.randint(1, 5)
        else:
            # Création d'un nouveau nœud avec deux branches
            arbre["branches"][f"Branche {i}"] = creer_arbre(genres_disponibles, probabilite_feuille)

    return arbre


# Fonction pour calculer le score final de l'arbre
def calculer_score(arbre):
    if isinstance(arbre, dict):
        score_total = 0
        for i in range(2):
            score_branche = calculer_score(arbre["branches"][f"Branche {i}"])
            if i == 0:
                score_total += (1 - arbre["score"]) * score_branche
            else:
                score_total += arbre["score"] * score_branche
        return score_total
    else:
        return arbre

def generer_matrice_scores(n_utilisateurs, n_films, genres, probabilite_feuille):
    scores_matrice = np.zeros((n_utilisateurs, n_films))  # Initialisation de la matrice de scores

    for u in range(n_utilisateurs):
        for f in range(n_films):
            # Générer un arbre aléatoire
            arbre = creer_arbre(genres.copy(), probabilite_feuille)
            # Calculer le score final de l'arbre
            score_final = calculer_score(arbre)
            # Ajouter le score final à la matrice
            scores_matrice[u, f] = score_final
    return scores_matrice

# --------------------------------------- DOUDA ---------------------------------- #

# Generate the complete matrix of ratings (oracle)
def generate_oracle(num_users,num_movies):
  # Create the movie genres categories
  genres = ["Action", "Comedy", "Drama", "Horror", "Sci-Fi"]
  # Assign preferences to users
  user_prefs = {genre: np.random.uniform(1, 5, num_users) for genre in genres}
  # Assign genres to films (maximum 3)
  film_genres = {f"Film {i}": np.random.choice(genres, np.random.randint(1, len(genres)-1), replace=False) for i in range(num_movies)}
  # Generate ratings based on the users preferences and genres
  oracle = np.zeros((num_users, num_movies))
  for user in range(num_users):
      for film in range(num_movies):
          film_genres_list = film_genres[f"Film {film}"]
          user_prefs_for_film = [user_prefs[genre][user] for genre in film_genres_list]
          avg_user_pref = np.mean(user_prefs_for_film)  # Average(Mean) of preferences to each movie genre
          oracle[user, film] = avg_user_pref
  return oracle


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


# Oracle settings
num_users = 943
num_movies = 1682


# Liste de genres
genres = ["Action", "Comédie", "Drame", "Science-Fiction", "Horreur"]

# Probabilité de tomber sur une feuille (note entre 1 et 5)
probabilite_feuille = 0.3

scores_matrice = generer_matrice_scores(num_users, num_movies, genres, probabilite_feuille)
movielens_format_oracle = convert_to_movielens_format(scores_matrice) # rinted values

# Generate a complete oracle
#oracle = generate_oracle(num_users,num_movies)
#movielens_format_oracle = convert_to_movielens_format(oracle) # rinted values


# Set the oracle to use
#df = pd.read_csv("movielens100k.csv")
df = movielens_format_oracle


# Set the testing parameters
densities = [0.9,0.8,0.7]
noise_levels = [0.0,0.5,1.0]
models = ["SVD++", "KNN", "SVD (ALS)"]

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

# Limiter les valeurs de l'axe x aux densités utilisées
x_ticks_labels = [f"{density_percentage*100}% ({int(len(df) * density_percentage)})" for density_percentage in densities]
plt.xticks([int(len(df) * density_percentage) for density_percentage in densities], x_ticks_labels)

plt.show()