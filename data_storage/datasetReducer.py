import logging
import time

from DBManager import DBManager


class DatasetReducer(DBManager):
    def __init__(self):
        super().__init__()

    def filterTracks(self, n):
        statement = '''DELETE FROM track
                        WHERE track_uri IN (SELECT playlist_track.track_uri
                        FROM playlist_track
                        GROUP BY playlist_track.track_uri
                        HAVING COUNT(*) <= %s)'''
        with self.getCursor() as cur:
            cur.execute(statement, n)
            self._conn.commit()

    def filterArtists(self, n):
        statement = '''DELETE FROM artist
                        WHERE artist_uri IN (SELECT artist.artist_uri
                        FROM playlist_track
                        NATURAL JOIN track
                        NATURAL JOIN album
                        NATURAL JOIN artist
                        GROUP BY artist.artist_uri
                        HAVING COUNT(*) <= 3)'''
        with self.getCursor() as cur:
            cur.execute(statement)
            self._conn.commit()


'''
artist 207724
track 1477959 -> 304978
playlist_track 27,482,811 -> 25,520,198
'''
if __name__ == '__main__':
    # TIMER START
    start = time.time()

    r = DatasetReducer()
    r.filterTracks(n=5)
    r.filterArtists(n=3)

    # TIMER DONE
    end = time.time()
    logging.info(str(end - start) + " seconds")
