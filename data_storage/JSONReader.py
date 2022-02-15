import json
import logging
import os
import time
from typing import Dict, Tuple
import sys
import config

from psycopg2.extras import execute_values

from DBManager import DBManager

logging.getLogger().setLevel(logging.INFO)


class JSONReader(DBManager):
    def __init__(self, dir_path : str, n=1000):
        super().__init__()
        self._dir_path = dir_path
        self._n = n  # number of files in directory to parse

    def start(self):
        if not self._dir_path:
            logging.error("Check environmental variable for DATASET_PATH")
            raise Exception("No path to directory provided.")

        total = len(os.listdir(self._dir_path)[:self._n])
        for i, file in enumerate(os.listdir(self._dir_path)[:self._n]):
            if file.endswith(".json"):
                self.insertIntoDB(*self.parse("%s/%s" % (self._dir_path, file)))
                self.printProgressBar(i, total)


    def parse(self, file_path : str):
        '''
        Given the path to the json file, extract all the artists, albums, tracks and playlists into dictionaries.
        Storing all the values in dictionaries acts as a buffer so that all values can be inserted at once,
        this will mitigate the latency of the database connection.
        '''
        artists: Dict[str, str] = {}  # Dictionaries enforce unique keys
        albums: Dict[str, Tuple] = {}
        tracks: Dict[str, Tuple] = {}
        playlist_tracks: Dict[Tuple, int] = {}
        playlists: Dict[int, Tuple] = {}

        open_file = open(file_path)
        data = json.load(open_file)
        if data['info'] and data['playlists']:
            for p in data['playlists']:
                playlists[p['pid']] = (p['name'], p['collaborative'], p['modified_at'], p['num_tracks'], p['num_albums'],
                                       p['num_followers'], p['num_edits'], p['duration_ms'], p['num_artists'])
                for t in p['tracks']:
                    artists[t['artist_uri']] = t['artist_name']
                    albums[t['album_uri']] = (t['album_name'], t['artist_uri'])
                    tracks[t['track_uri']] = (t['track_name'], t['duration_ms'], t['album_uri'])
                    playlist_tracks[(t['track_uri'], p['pid'])] = t['pos']

        return artists, albums, tracks, playlist_tracks, playlists

    def insertIntoDB(self, artists, albums, tracks, playlist_tracks, playlists):
        with self.getCursor() as cur:
            insert_artists = '''INSERT INTO artist (artist_uri, artist_name)
                                VALUES %s ON CONFLICT DO NOTHING;'''
            execute_values(cur, insert_artists, list(artists.items()))

            insert_albums = '''INSERT INTO album (album_uri, album_name, artist_uri)
                               VALUES %s ON CONFLICT DO NOTHING;'''
            execute_values(cur, insert_albums, [(x, *albums[x]) for x in albums])

            insert_tracks = '''INSERT INTO track (track_uri, track_name, duration, album_uri)
                               VALUES %s ON CONFLICT DO NOTHING;'''
            execute_values(cur, insert_tracks, [(x, *tracks[x]) for x in tracks])

            insert_playlists = '''INSERT INTO playlist (pid, playlist_name, collaborative, modified, num_tracks,
                                  num_albums, num_followers, num_edits, duration_ms, num_artists)
                                  VALUES %s ON CONFLICT DO NOTHING;'''
            execute_values(cur, insert_playlists, [(x, *playlists[x]) for x in playlists])

            insert_playlist_tracks = '''INSERT INTO playlist_track (track_uri, pid, pos) 
                                        VALUES %s ON CONFLICT DO NOTHING;'''
            execute_values(cur, insert_playlist_tracks, [(*x, playlist_tracks[x]) for x in playlist_tracks])
            self._conn.commit()

    def printProgressBar(self, i, max):
        n_bar = 30  # size of progress bar
        j = i / max
        sys.stdout.write('\r')
        sys.stdout.write(f"[{'=' * int(n_bar * j):{n_bar}s}] {int(100 * j)}%")
        sys.stdout.flush()


if __name__ == '__main__':
    # TIMER START
    start = time.time()

    j = JSONReader(config.DATASET_PATH)
    j.start()

    # TIMER DONE
    end = time.time()
    logging.info(str(end - start) + " seconds")
