import csv
import pickle
import torch
import numpy as np
import json
from tqdm import tqdm
import gc
from cbow_model.model import load_model

# ADJUSTABLE PARAMETERS THAT CHANGE RUN SPEED
MINIMUM_FREQ_THRESHOLD = 50  # the minimum number of playlists a song has to be in to be considered for model output
N_TRACKS = 1_000_000  # subset of playlist to train on - maximum of 1,000,000
USE_GENSIM = False  # Dictates which song embeddings to used. true means gensim are used, false means ours are used.
BACKUP_AVAILABLE = False  # If a backup file of the dataframe has been created set to true otherwise false

GENSIM_PKL = r"C:\Users\liamb\Documents\Spotify-Million-Playlist-Challenge\song_embeddings.pkl"

CBOW_MODEL_PATH = r"C:\Users\liamb\Documents\Spotify-Million-Playlist-Challenge\CBOW_run_1M_min_5_PP@2024-04-26-10-27-43_con5_pl1000000_emb64_ep1.pt"
CBOW_MAPPING = r"C:\Users\liamb\Documents\Spotify-Million-Playlist-Challenge\CBOW_run_1M_min_5_PP@2024-04-26-10-27-43_con5_pl1000000_emb64_ep1-track2idx.pkl"

CHALLENGE_SET_PATH = r"C:\Users\liamb\Documents\Spotify-Million-Playlist-Challenge\challenge_set.json"
CSV_OUT_PATH = r"C:\Users\liamb\Documents\Spotify-Million-Playlist-Challenge\cbow_INF.csv"

if USE_GENSIM:
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
else:
    # Load song embeddings
    print("Loading our embeddings...")
    model, track2idx = load_model(CBOW_MODEL_PATH, CBOW_MAPPING, context_size=5, embedding_dim=64)
    track_uris = np.array(list(track2idx.keys()))
    embeddings_matrix = model.embedding.weight.detach().numpy()
    global_average_embedding = np.mean(embeddings_matrix, axis=0)


print("Loading challenge dataset...")
# Load the JSON data from the file
with open(CHALLENGE_SET_PATH, 'r') as f:
    playlists_data = json.load(f)
all_playlists = playlists_data['playlists']



team_info = ["team_info", "soton", "lhb1g20@soton.ac.uk"]
with open(CSV_OUT_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(team_info)


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
            if USE_GENSIM:
                embedding = embeddings_dict.get(track_dict['track_uri'])
                if embedding is not None:
                    track_embeddings.append(embedding)
            else:
                idx = track2idx.get(track_dict['track_uri'])
                if idx is not None:
                    embedding = embeddings_matrix[idx]
                    track_embeddings.append(torch.tensor(embedding))

        if len(track_embeddings) > 0:
            average_embedding = np.mean(torch.stack(track_embeddings).numpy(), axis=0)

    if average_embedding is None:
        average_embedding = global_average_embedding

    # Filter the embedding matrix to not include the tracks in the playlist (tracks)
    idxs_to_ignore = [
        track2idx.get(track['track_uri'])
        for track in tracks
        if track2idx.get(track['track_uri']) is not None
    ]

    filtered_embeddings = np.delete(
        embeddings_matrix,
        idxs_to_ignore,
        axis=0
    )
    filtered_uris = np.delete(
        track_uris,
        idxs_to_ignore,
        axis=0
    )

    scores = np.dot(filtered_embeddings, average_embedding)
    top_500_indices = np.argpartition(scores, 500)[:500]

    output_line = [playlist_pid] + filtered_uris[top_500_indices].tolist()
    playlist_predictions.append(output_line)

    if len(playlist_predictions) == 2500:
        # Append the output line to the CSV file
        with open(CSV_OUT_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(playlist_predictions)
            playlist_predictions = []
