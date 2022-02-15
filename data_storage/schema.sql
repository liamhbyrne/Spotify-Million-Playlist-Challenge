CREATE TABLE IF NOT EXISTS artist (
    artist_uri  VARCHAR(50),
    artist_name TEXT,
    PRIMARY KEY (artist_uri)
);

CREATE TABLE IF NOT EXISTS album (
    album_uri VARCHAR(50),
    album_name TEXT,
    artist_uri VARCHAR(50) REFERENCES artist(artist_uri) ON DELETE CASCADE,
    PRIMARY KEY (album_uri)
);

CREATE TABLE IF NOT EXISTS track (
    track_uri VARCHAR(50),
    track_name TEXT,
    duration INTEGER,
    album_uri VARCHAR(50) REFERENCES album(album_uri) ON DELETE CASCADE,
    PRIMARY KEY (track_uri)
);

CREATE TABLE IF NOT EXISTS playlist (
    pid INTEGER,
    playlist_name TEXT,
    collaborative VARCHAR(10),
    modified INTEGER,
    num_tracks INTEGER,
    num_albums INTEGER,
    num_followers INTEGER,
    num_edits INTEGER,
    duration_ms INTEGER,
    num_artists INTEGER,
    PRIMARY KEY (pid)
);

CREATE TABLE IF NOT EXISTS playlist_track (
    track_uri VARCHAR(50) REFERENCES track (track_uri) ON DELETE CASCADE,
    pid INTEGER REFERENCES playlist (pid) ON DELETE CASCADE,
    pos INTEGER,
    PRIMARY KEY (track_uri, pid)
);