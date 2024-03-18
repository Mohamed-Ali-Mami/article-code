import numpy as np

def generate_factorized_matrix(num_users, num_items, rank=2):

    # Initialize user (A) and item (B) matrices with random values
    A = np.random.rand(num_users, rank)
    B = np.random.rand(rank, num_items)

    # Calculate the product to get the rating matrix
    ratings_matrix = np.dot(A, B)

    # Scale the rating matrix to have values between 1 and 5
    min_val, max_val = ratings_matrix.min(), ratings_matrix.max()
    scaled_ratings_matrix = 1 + 4 * (ratings_matrix - min_val) / (max_val - min_val)
    ratings_matrix = np.rint(scaled_ratings_matrix)  # Round to nearest integer

    return ratings_matrix

