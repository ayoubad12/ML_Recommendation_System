import streamlit as st
from demographic import demographic_filtering
from content_based import content_based_filtering
from collaborative import collaborative_filtering

# Sidebar menu
st.sidebar.title('Select Algorithm')

# List of algorithms
algorithms = ['Demographic Filtering', 'Content Based Filtering', 'Collaborative Filtering']

# Display radio buttons for each algorithm
selected_algorithm = st.sidebar.radio('Select Algorithm', algorithms)

# Main content area
if selected_algorithm == 'Demographic Filtering':
    demographic_filtering()
elif selected_algorithm == 'Content Based Filtering':
    content_based_filtering()
elif selected_algorithm == 'Collaborative Filtering':
    collaborative_filtering()
