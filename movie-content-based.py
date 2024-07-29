import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import re

# Load your movies dataframe
movies_df = pd.read_csv("movies.csv")

# Replace '|' with space to create a bag of words format
movies_df['genres'] = movies_df['genres'].str.replace('|', ' ')

# Convert genres to a matrix of token counts
count_vectorizer = CountVectorizer()
genre_matrix = count_vectorizer.fit_transform(movies_df['genres'])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

# Function to prioritize sequels
def prioritize_sequels(recommendations, original_title, sim_scores):
    original_base_title = re.sub(r'\(\d{4}\)', '', original_title).strip().lower()

    # Function to identify if a movie is a sequel of the original movie
    def is_sequel(movie_title):
        base_title = re.sub(r'\(\d{4}\)', '', movie_title).strip().lower()
        return base_title.startswith(original_base_title) and base_title != original_base_title

    # Separate sequels and non-sequels
    sequels = [(title, score) for title, score in zip(recommendations, sim_scores) if is_sequel(title)]
    non_sequels = [(title, score) for title, score in zip(recommendations, sim_scores) if not is_sequel(title)]

    # Determine the highest score among non-sequels
    max_non_sequelscore = max(score for _, score in non_sequels) if non_sequels else 0

    # Filter sequels with scores greater than or equal to the highest non-sequel score
    filtered_sequels = [title for title, score in sequels if score >= max_non_sequelscore]
    non_sequels = [title for title, score in non_sequels]

    # Return sequels first, followed by non-sequels
    return filtered_sequels + non_sequels

# Function to get recommendations
def get_recommendations(title, num_rec=10, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = movies_df[movies_df['title'] == title].index[0]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the num_rec most similar movies (plus one for the input movie)
    sim_scores = sim_scores[:num_rec + 2]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Get the titles and similarity scores of the top num_rec most similar movies
    recommendations = movies_df['title'].iloc[movie_indices].tolist()
    sim_scores = [i[1] for i in sim_scores]

    # Remove the target movie from the recommendations
    recommendations = [rec for rec in recommendations if rec != title]
    sim_scores = sim_scores[1:]  # Remove the first score (which corresponds to the input movie)

    # Prioritize sequels
    recommendations = prioritize_sequels(recommendations, title, sim_scores)

    # Adjust the number of recommendations to the available size
    num_rec = min(num_rec, len(recommendations))

    # Return only the requested number of recommendations
    return recommendations[:num_rec]

# Streamlit app
st.title("Movie Recommendation System")

# Custom CSS to style the subheader, button, and image
st.markdown("""
    <style>
    .custom-subheader {
        font-size: 20px;
    }
    .image-container {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white !important;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
        color: white !important;
    }
    .stButton > button:active {
        background-color: #f44336; /* Red color when clicked */
        color: white !important;
    }
    .stButton > button:focus {
        background-color: #4CAF50;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# Subheader with custom size
st.markdown('<p class="custom-subheader">Find Your Next Watch Based on "Genre" Similarities</p>', unsafe_allow_html=True)

# Center the image
st.markdown("""
    <div class="image-container">
        <img src="https://lajoyalink.com/wp-content/uploads/2018/03/Movie.jpg" width="350">
    </div>
""", unsafe_allow_html=True)

# Autocomplete for movie title
title = st.selectbox(
    "Enter a movie title:",
    options=[""] + movies_df['title'].tolist(),
    index=0,
    key="movie_title",
    format_func=lambda x: x if x else "",
)

# Input number of recommendations
num_rec_input = st.text_input("Number of recommendations:", value="10")

# Get recommendations
if st.button("Get Recommendations", key="recommendations"):
    if title.strip() == "":
        st.error("Please select a movie title.")
    else:
        try:
            num_rec = int(num_rec_input)
            if num_rec < 1:
                raise ValueError
            recommendations = get_recommendations(title, num_rec)
            st.write(f"Top {len(recommendations)} recommendations for '{title}':")
            for i, movie in enumerate(recommendations, 1):
                st.write(f"{i}. {movie}")
        except ValueError:
            st.error("Please enter a valid number.")
