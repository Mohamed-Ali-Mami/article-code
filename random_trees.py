import random
import numpy as np

# Function to create a random tree with a specific leaf probability and genre options
def create_tree(genres_list, leaf_probability=0.5):
    if not genres_list or random.random() < leaf_probability:
        # if the genre list is empty or the leaf probability is reached, return a random rating
        return random.randint(1, 5)

    # Random choice of a movie genre
    genre = random.choice(genres_list)
    genres_list.remove(genre)

    # Creation of a random score for the randomly chosen genre
    genre_score = round(random.uniform(0, 1), 1)  # Random score between 0 et 1.

    # Creation of a node with the randomly chosen genre and score
    tree = {"genre": genre, "score": genre_score, "branches": {}}

    # Creation of 2 branches for each node
    for i in range(2):
        # Random choise between a rating or a new node respecting the leaf probability
        if random.random() < leaf_probability:
            # Creation of a random rating between 1 and 5
            tree["branches"][f"Branch {i}"] = random.randint(1, 5)
        else:
            # Creation of a new node
            tree["branches"][f"Branch {i}"] = create_tree(genres_list, leaf_probability)
    # Return the final tree
    return tree


# Calculate the final rating of a tree
def calculate_tree_rating(tree):
    if isinstance(tree, dict):
        final_rating = 0
        for i in range(2):
            score_branch = calculate_tree_rating(tree["branches"][f"Branch {i}"])
            if i == 0:
                final_rating += (1 - tree["score"]) * score_branch
            else:
                final_rating += tree["score"] * score_branch
        return final_rating
    else:
        return tree


# Function to generate a matrix containing ratings of users with our tree method
def generate_ratings_matrix(num_users, num_movies, genres, probabilite_feuille):
    ratings_matrix = np.zeros((num_users, num_movies))  # Initialize the matrix

    for u in range(num_users):
        for f in range(num_movies):
            # Generate random trees
            random_tree = create_tree(genres.copy(), probabilite_feuille)
            # Calculate the final rating of each tree
            score_final = calculate_tree_rating(random_tree)
            # Add the final rating to the matrix
            ratings_matrix[u, f] = score_final

    # Return the generated tree
    return ratings_matrix