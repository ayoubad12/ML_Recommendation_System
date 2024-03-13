import streamlit as st
from surprise import SVD, Reader, Dataset
import pandas as pd

# Load the model and data
reader = Reader()
ratings = pd.read_csv('my dataset/ratings_small.csv')
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
svd = SVD()
trainset = data.build_full_trainset()
svd.fit(trainset)

# Prediction function
def predict_rating(user_id, movie_id):
    prediction = svd.predict(user_id, movie_id, 3)
    return prediction.est

# Main UI code
def main():
    st.title('Movie Rating Prediction')
    user_id = st.text_input('Enter User ID:')
    movie_id = st.text_input('Enter Movie ID:')
    
    if st.button('Predict Rating'):
        try:
            user_id = int(user_id)
            movie_id = int(movie_id)
            rating = predict_rating(user_id, movie_id)
            st.success(f'The estimated rating for movie {movie_id} by user {user_id} is {rating:.2f}')
        except ValueError:
            st.error('Please enter valid user ID and movie ID')

if __name__ == '__main__':
    main()
