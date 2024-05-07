import csv
import pickle
import torch
import numpy as np
import json
from tqdm import tqdm
from gensim.models import Word2Vec

CHALLENGE_SET_PATH = r"C:\Users\Sam\spotify\challenge_set.json"
CSV_OUT_PATH = r"C:\Users\Sam\spotify\gensim.csv"

model = Word2Vec.load(r"C:\Users\Sam\Spotify-Million-Playlist-Challenge\track2vec2.model")

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
            track_uri = track_dict['track_uri']
            try:
                embedding = model.wv[track_uri]
                track_embeddings.append(embedding)
            except KeyError:
                continue

        if len(track_embeddings) > 1:
            average_embedding = np.mean(np.array(track_embeddings), axis=0)

    if average_embedding is None:
        # TODO: improve
        average_embedding = model.wv["spotify:track:7qiZfU4dY1lWllzX7mPBI3"]

    # Filter the embedding matrix to not include the tracks in the playlist (tracks)
    uris_to_ignore = {
        track['track_uri']
        for track in tracks
    }

    most_similar_tracks = model.wv.similar_by_vector(average_embedding, topn=500+len(tracks))
    recommended_tracks = []
    for track_uri, similarity in most_similar_tracks:
        if track_uri not in uris_to_ignore:
            recommended_tracks.append(track_uri)
    output_line = [playlist_pid] + recommended_tracks[:500]
    playlist_predictions.append(output_line)

    if len(playlist_predictions) == 2500:
        # Append the output line to the CSV file
        with open(CSV_OUT_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(playlist_predictions)
            playlist_predictions = []
