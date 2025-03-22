import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import ast
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from functools import lru_cache


st.set_page_config(page_title="Spotify Song Recommender", page_icon="🎵")


# define feature columns:
feature_columns = [
    'acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness',
    'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo',
    'time_signature', 'valence', 'mean_words_sentence', 'n_sentences', 'n_words',
    'spectral_centroid', 'spectral_bandwith', 'Chroma_1', 'Chroma_5', 'Chroma_8',
    'Spectral_contrast_1', 'Spectral_contrast_2', 'Spectral_contrast_3',
    'Tonnetz_1', 'Tonnetz_2', 'Tonnetz_3', 'Tonnetz_4', 'Tonnetz_5', 'Tonnetz_6',
    'MFCC_2', 'MFCC_3', 'MFCC_4', 'MFCC_5', 'MFCC_6', 'MFCC_7', 'MFCC_8',
    'MFCC_9', 'MFCC_10', 'MFCC_11', 'MFCC_12', 'MFCC_13',
    'ZCR', 'spectral_rollOff_min', 'entropy_energy'
]

# load data, cached to reduce computation time
@st.cache_data(show_spinner=False)
def load_data():
    spotify_db = pd.read_table("data/spotify_knn_features.csv", sep=",")
    spotify_db['artist_names'] = spotify_db['artist_names'].apply(ast.literal_eval)
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(spotify_db[feature_columns])
    
    # Save the scaled features as a NumPy array
    spotify_db['scaled_features'] = list(scaled_features)

    return spotify_db, scaler

spotify_db, scaler = load_data()
# load model once to reduce computation time
@st.cache_resource(show_spinner=False)
def load_recommender_model():
    return pickle.load(open("./model/spotify_recommendation_model.pkl", "rb"))

song_recommender = load_recommender_model()


# set dummy for embedding spotify songs on website
spotify_song_height = 200
@lru_cache(maxsize=1000)
def get_spotify_song_html(track_id: str):
    spotify_url = f"""
    <iframe style="border-radius:12px" 
    src="https://open.spotify.com/embed/track/{track_id}?utm_source=generator" 
    width="100%" 
    height="{spotify_song_height}" 
    frameBorder="0" 
    allowfullscreen="" 
    allow="autoplay; 
    clipboard-write; 
    encrypted-media; 
    fullscreen; 
    picture-in-picture" 
    loading="lazy">
    </iframe>
    """
    return spotify_url


# define song recommendation function
@st.cache_resource(show_spinner=False)
def recommend_songs_for(song_id: object, knn=song_recommender, k=5):
    song_index = spotify_db[spotify_db['id'] == song_id].index

    # if no song index was found return
    if len(song_index) == 0:
        return "Song not found!"

    # scale input features to match the input used for training.
    song_features = spotify_db.loc[song_index, feature_columns]
    input_song = scaler.transform(song_features)

    # Find nearest neighbours
    distances, indices = knn.kneighbors(input_song, n_neighbors=k+1)
    recommended_songs_ids = indices[0][1:]  # Excluding the first because that is the input itself
    
    recommended_songs = spotify_db.iloc[recommended_songs_ids][['id']]

    return recommended_songs



#  Search selection
search_mode = st.selectbox(
    "Search by:",
    options=["Artist", "Song", "Album"]
)

# Search input
search_term = st.text_input("Search for:")

# define function that searches song name, artists name and album name simultaneously and returns the results.
def matches_search(row, search_term):
    search_term = search_term.lower()
    if search_mode == "Artist":
        artist_match = any(search_term in artist.lower() for artist in row['artist_names'])
        return artist_match
    elif search_mode == "Album":
        album_match = search_term in row['album_name'].lower()
        return album_match
    elif search_mode == "Song":
        title_match = search_term in row['name'].lower()
        return title_match
    return False


if "num_display" not in st.session_state:
    st.session_state.num_display = 10  # Show 10 results at first

# Initialize session state
if "selected_song_id" not in st.session_state:
    st.session_state.selected_song_id = None

# if song is currently selected
if st.session_state.selected_song_id:
    st.write("A song is selected:")
    components.html(get_spotify_song_html(st.session_state.selected_song_id), height=spotify_song_height)

    # Clear selection button
    if st.button("Clear selection"):
        st.session_state.selected_song_id = None
        st.rerun()  # Refresh the app so other songs show again

    st.write("Recommendations based on the song:")
    recommendations = recommend_songs_for(st.session_state.selected_song_id) # create similar song recommendations
    for song_id in recommendations["id"]: # list recommendations
        col1, col2 = st.columns([3, 1])
        with col1:
            components.html(get_spotify_song_html(song_id), height=spotify_song_height)
        with col2:
            if st.button("Select", key=song_id):
                st.session_state.selected_song_id = song_id
                st.rerun()

else: # else, search for songs
    # Search and filtering
    if search_term:
        if search_mode == "Artist":
            mask = spotify_db['artist_names'].apply(lambda artists: any(search_term.lower() in artist.lower() for artist in artists))
        elif search_mode == "Album":
            mask = spotify_db['album_name'].str.lower().str.contains(search_term.lower())
        elif search_mode == "Song":
            mask = spotify_db['name'].str.lower().str.contains(search_term.lower())
        filtered_df = spotify_db[mask].sort_values(by=['album_name', 'album_id', 'name'])

        # Only display list if no song is selected
        if st.session_state.selected_song_id is None:
            for index, row in filtered_df.iloc[:st.session_state.num_display].iterrows():
                col1, col2 = st.columns([3, 1])
                with col1:
                    components.html(get_spotify_song_html(row["id"]), height=spotify_song_height)
                with col2:
                    if st.button("Select", key=row['id']):
                        st.session_state.selected_song_id = row["id"]
                        st.rerun()
            if len(filtered_df) > st.session_state.num_display:
                if st.button("Load more results"):
                    st.session_state.num_display += 10
                    st.rerun()
    
        else:
            # Display only the selected song
            row = st.session_state.selected_song
            st.success(f"Selected: {row['name']} by {', '.join(row['artist_names'])}")
            components.html(get_spotify_song_html(row["id"]), height=spotify_song_height)

