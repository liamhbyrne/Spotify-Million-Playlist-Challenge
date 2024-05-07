from tqdm import tqdm
import sqlite3
from gensim.models import Word2Vec
import polars as pl
from gensim.models.callbacks import CallbackAny2Vec
import csv
import logging
import os
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


logging.info("Creating dataset")
if not os.path.exists("playlists.cor"):
    conn = sqlite3.connect(r'C:\Users\Sam\spotify\spotify.db')
    query = '''
    SELECT PT.pid, GROUP_CONCAT(PT.track_uri)
    FROM playlist_track PT
    GROUP BY PT.pid
    '''
    playlist_tracks = pl.read_database(query, conn)
    playlists = [row.split(",") for row in tqdm(playlist_tracks["GROUP_CONCAT(PT.track_uri)"])]
    with open('playlists.cor', 'w', newline='') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(playlists)
    del playlist_tracks
    del playlists
    conn.close()


class MonitorCallback(CallbackAny2Vec):
    def on_epoch_end(self, model):
        print("Model loss:", model.get_latest_training_loss())
        # The Taylor Swift: Love Story test
        conn = sqlite3.connect(r'C:\Users\Sam\spotify\spotify.db')
        print("The 25 most similar songs to Taylor Swift's Love Story are:")
        for i, (track_uri, similarity) in enumerate(model.wv.most_similar("spotify:track:1vrd6UOGamcKNGnSHJQlSt", topn=50)):
            query = f"""
                SELECT track.track_name, artist.artist_name
                FROM track
                JOIN album on track.album_uri = album.album_uri
                JOIN artist ON album.artist_uri = artist.artist_uri
                WHERE track.track_uri == '{track_uri}';
                """
            track_df = pl.read_database(query, conn)
            print(f" {i}. {similarity:.3f} - {track_df['artist_name'][0]}: {track_df['track_name'][0]}")

logging.info("Training the model")
model = Word2Vec(corpus_file='playlists.cor', vector_size=256, window=5, min_count=25, sg=1, callbacks=[MonitorCallback()])
model.save("track2vec2.model")
