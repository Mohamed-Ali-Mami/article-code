import numpy as np

# Generate the complete matrix of ratings (oracle)
def generate_oracle(num_users,num_movies,genres):
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