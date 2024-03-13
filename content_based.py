import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the datasets
df1 = pd.read_csv('my dataset/tmdb_5000_credits.csv')
df2 = pd.read_csv('my dataset/tmdb_5000_movies.csv')
df1.columns = ['id', 'title', 'cast', 'crew']   
df2 = df2.merge(df1, on='id')

def content_based_filtering():
    tfidf = TfidfVectorizer(stop_words='english')
    # Replace NaN with an empty string
    df2['overview'] = df2['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(df2['overview'])

    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Construct a reverse map of indices and movie titles
    # indices = pd.Series(df2.index, index=df2['original_title'].str.lower().str.strip()).drop_duplicates()
    indices = pd.Series(df2.index, index=df2['original_title']).drop_duplicates()

    # Function that takes in movie title as input and outputs most similar movies
    def get_recommendations(title, cosine_sim=cosine_sim):
        # Get the index of the movie that matches the title
        idx = indices[title]
        # Get the pairwsie similarity scores of all movies with that movie
        sim_scores = list(enumerate(cosine_sim[idx]))
        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Get the scores of the 10 most similar movies
        sim_scores = sim_scores[1:11]
        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]
        # Return the top 10 most similar movies
        return df2['original_title'].iloc[movie_indices]

    # Streamlit UI
    st.title('Content-Based Movie Recommendation System')
    movie_title = st.text_input('Enter a movie title:')
    if st.button('Get Recommendations'):
        try:
            recommended_movies = get_recommendations(movie_title)
            st.write('Top 10 Recommended Movies:')
            st.write('* ' + '\n* '.join(recommended_movies))
        except KeyError:
            st.error('Movie not found. Please enter a valid movie title.')

#content_based_filtering()
