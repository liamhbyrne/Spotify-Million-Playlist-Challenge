
import csv
import difflib
import math
import json
from typing import List

from tqdm import tqdm
from gensim.models import Word2Vec
import polars as pl
import sqlite3

CHALLENGE_SET_PATH = r"C:\Users\Sam\spotify\challenge_set.json"
CSV_OUT_PATH = r"C:\Users\Sam\spotify\gensim_n_songs_playlist_sim.csv"

model = Word2Vec.load(r"C:\Users\Sam\Spotify-Million-Playlist-Challenge\word2vec.model")

print("Loading challenge dataset...")
# Load the JSON data from the file
with open(CHALLENGE_SET_PATH, 'r') as f:
    playlists_data = json.load(f)
all_playlists = playlists_data['playlists']

team_info = ["team_info", "soton", "lhb1g20@soton.ac.uk"]
with open(CSV_OUT_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(team_info)

print("Loading playlist name mapping")
conn = sqlite3.connect(r'C:\Users\Sam\spotify\spotify.db')
query = f"""
    SELECT playlist.pid, playlist.playlist_name
    FROM playlist
    LIMIT 5000
    """
playlist_df = pl.read_database(query, conn)

def get_closest_pids(playlist_name: str, n=1) -> List[int]:
    closest_playlist_names = difflib.get_close_matches(
        playlist_name, playlist_df["playlist_name"].to_list(),
        n=n, cutoff=0
    )
    print(f"Matched {playlist_name} to {closest_playlist_names}")
    return playlist_df.filter(pl.col("playlist_name").is_in(closest_playlist_names))["pid"].to_list()

def get_tracks_of_similar_playlist(playlist_name: str) -> List[str]:
    pids = get_closest_pids(playlist_name)
    query = f"""
        SELECT PT.pid, GROUP_CONCAT(PT.track_uri)
        FROM playlist_track PT
        WHERE PT.pid == {pids[0]}
        GROUP BY PT.pid;
        """
    playlist_tracks = pl.read_database(query, conn)
    return playlist_tracks[0]["GROUP_CONCAT(PT.track_uri)"].item().split(',')
    # return playlist_tracks.filter(
    #     pl.col("GROUP_CONCAT(PT.track_uri)").str.len_bytes() == playlist_tracks["GROUP_CONCAT(PT.track_uri)"].str.len_bytes().max()
    # )["GROUP_CONCAT(PT.track_uri)"].item().split(',')


playlist_predictions = []
for playlist in tqdm(all_playlists):
    # Extract playlist name and pid
    playlist_name = playlist.get('name', None)
    playlist_pid = playlist.get('pid', None)
    n_tracks = playlist.get('num_tracks', 0)
    tracks = playlist['tracks']

    # If there are >0 tracks, calculate the average song embedding
    track_uris = []
    for track_dict in tracks:
        track_uri = track_dict['track_uri']
        try:
            _ = model.wv[track_uri]
            track_uris.append(track_uri)
        except KeyError:
            continue

    if len(track_uris) <= 5:
        # If the playlist contains few known songs, find the playlist with the most
        # similar name and use those songs
        print(f"track_uris {len(track_uris)}")
        similar_track_uris = get_tracks_of_similar_playlist(playlist_name)
        for track_uri in similar_track_uris:
            try:
                _ = model.wv[track_uri]
                if track_uri not in track_uris:
                    track_uris.append(track_uri)
            except KeyError:
                continue
        print(f"augmented to {len(track_uris)}")

        # If it's still zero, use Shape of You
        if len(track_uris) == 0:
            print("track uris still empty... using Shape of You...")
            track_uris = ["spotify:track:7qiZfU4dY1lWllzX7mPBI3"]

    # Filter the embedding matrix to not include the tracks in the playlist (tracks)
    uris_to_ignore = {
        track['track_uri']
        for track in tracks
    }

    recommended_tracks = []
    n_per_track = math.ceil(500 / max(len(track_uris), 1))
    for track_embedding in track_uris:
        most_similar_tracks = model.wv.most_similar(track_embedding, topn=500)
        n_from_track = 0
        for track_uri, similarity in most_similar_tracks:
            if track_uri not in uris_to_ignore:
                recommended_tracks.append((track_uri, similarity))
                uris_to_ignore.add(track_uri)
                n_from_track += 1
            if n_from_track >= n_per_track:
                break

    if len(recommended_tracks) < 500:
        print(f"There are {len(recommended_tracks)}, using worse recommendations")
        most_similar_tracks = model.wv.most_similar(track_uris[-1], topn=500+len(tracks))
        for track_uri, similarity in most_similar_tracks:
            if track_uri not in uris_to_ignore:
                recommended_tracks.append((track_uri, similarity))
                uris_to_ignore.add(track_uri)

    # Sort recommended tracks by similarity
    recommended_tracks = sorted(recommended_tracks, key=lambda x: x[1], reverse=True)
    tracks = [track for track, _ in recommended_tracks]
    output_line = [playlist_pid] + tracks[:500]
    playlist_predictions.append(output_line)

    if len(playlist_predictions) == 2500:
        # Append the output line to the CSV file
        with open(CSV_OUT_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(playlist_predictions)
            playlist_predictions = []
