import logging
import time

from DBManager import DBManager


class DatasetReducer(DBManager):
    """
    This class is used to reduce the size of the dataset by removing tracks and artists that appear in less than n
    playlists.
    """

    def __init__(self):
        super().__init__()

    def filter_tracks(self, n):
        """
        Remove all tracks that appear in less than n playlists.
        """
        statement = f"""DELETE FROM track
                        WHERE track_uri IN (SELECT playlist_track.track_uri
                        FROM playlist_track
                        GROUP BY playlist_track.track_uri
                        HAVING COUNT(*) <= {n})"""
        cursor = self.get_cursor()
        cursor.execute(statement)
        self._conn.commit()

    def filter_artists(self, n):
        """
        Remove all artists that appear in less than n playlists.

        NOTE: Not tested, check before and after running.
        """
        statement = f"""DELETE FROM artist
                       WHERE artist_uri IN (
                           SELECT artist.artist_uri
                           FROM playlist_track
                           JOIN track ON playlist_track.track_uri = track.track_uri
                           JOIN album ON track.album_uri = album.album_uri
                           JOIN artist ON album.artist_uri = artist.artist_uri
                           GROUP BY artist.artist_uri
                           HAVING COUNT(*) <= {n}
                       )"""

        cursor = self.get_cursor()
        cursor.execute(statement)
        self._conn.commit()


"""
artist 207,724
track 1,477,959 -> 30,4978
playlist_track 27,482,811 -> 25,520,198
"""
if __name__ == "__main__":
    # TIMER START
    start = time.time()

    r = DatasetReducer()
    r.filter_tracks(n=5)
    r.filter_artists(n=100000000)

    # TIMER DONE
    end = time.time()
    logging.info(str(end - start) + " seconds")
