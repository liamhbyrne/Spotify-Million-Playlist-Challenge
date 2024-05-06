import csv
import pickle
import torch
import numpy as np
import json
from tqdm import tqdm

# ADJUSTABLE PARAMETERS THAT CHANGE RUN SPEED
MINIMUM_FREQ_THRESHOLD = 50  # the minimum number of playlists a song has to be in to be considered for model output
N_TRACKS = 1_000_000  # subset of playlist to train on - maximum of 1,000,000
USE_GENSIM = True  # Dictates which song embeddings to used. true means gensim are used, false means ours are used.
BACKUP_AVAILABLE = False  # If a backup file of the dataframe has been created set to true otherwise false

GENSIM_PKL = r"C:\Users\theaw\spotify\song_embeddings.pkl"
# PYTORCH_PKL = "/mainfs/lyceum/lhb1g20/Spotify-Million-Playlist-Challenge/cbow_model/model_states/CBOW_run_1M_min_5_PP@2024-04-26-10-27-43_con5_pl1000000_emb64_ep1-track2idx.pkl"
# PYTORCH_MODEL = "/mainfs/lyceum/lhb1g20/Spotify-Million-Playlist-Challenge/cbow_model/model_states/CBOW_run_1M_min_5_PP@2024-04-26-10-27-43_con5_pl1000000_emb64_ep1.pt"

DB_LOCATION = r"C:\Users\theaw\spotify\spotify.db"
CHALLENGE_SET_PATH = r"C:\Users\theaw\spotify\challenge_set.json"
CSV_OUT_PATH = r"C:\Users\theaw\Spotify-Million-Playlist-Challenge\playlist_predictions.csv"


# Load song embeddings
print("Loading gensim embeddings...")
with open(GENSIM_PKL, 'rb') as f:
    embeddings_dict = pickle.load(f)
# track_uri: embedding
for key, value in embeddings_dict.items():
    embeddings_dict[key] = torch.tensor(value)

print("Creating track_uri and embedding matrices...")
track_uris, embeddings = zip(*embeddings_dict.items())
track_uris = np.array(track_uris)
embeddings_matrix = np.stack([value.numpy() for value in embeddings])
global_average_embedding = np.mean(embeddings_matrix, axis=0)

print("Loading challenge dataset...")
# Load the JSON data from the file
with open(CHALLENGE_SET_PATH, 'r') as f:
    playlists_data = json.load(f)
all_playlists = playlists_data['playlists']

playlist_predictions = []
for playlist in tqdm(all_playlists):
    # Extract playlist name and pid
    playlist_name = playlist.get('name', None)
    playlist_pid = playlist.get('pid', None)
    n_tracks = playlist.get('num_tracks', 0)
    tracks = playlist['tracks']

    # If there are >0 tracks, calculate the average song embedding
    average_embedding = None
    if n_tracks > 0:
        track_embeddings = []
        for track_dict in tracks:
            embedding = embeddings_dict.get(track_dict['track_uri'])
            if embedding is not None:
                track_embeddings.append(embedding)
            else:
                continue  # If embedding does not exist, ignore it
        if track_embeddings:
            average_embedding = np.mean(track_embeddings, axis=0)
    if not average_embedding:  # n_tracks==0 or no known tracks
        # Use the average embedding of a playlist with the most similar name.
        # TODO
        average_embedding = global_average_embedding

    scores = np.dot(embeddings_matrix, average_embedding)
    top_500_indices = np.argpartition(scores, 500)[500:]
    output_line = [playlist_pid] + track_uris[top_500_indices].tolist()
    playlist_predictions.append(output_line)

# Write the playlist predictions to a CSV file
with open(CSV_OUT_PATH, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(playlist_predictions)
