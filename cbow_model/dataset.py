import polars as pl
import torch
from torch.utils.data import Dataset
from typing import List

from data_storage.DBManager import DBManager


class CBOWDataset(Dataset):
    """
    Selects tracks from the dataset on the SQLite database.
    """
    def __init__(self, db_manager: DBManager, n_playlists: int, context_size: int = 5):
        self.n_playlists = n_playlists
        self.context_size = context_size
        self.db = db_manager
        self.cur = self.db.get_cursor()
        self.dataset = self._make_dataset()

    def get_named_tracks(self) -> List[str]:
        return [
            self.dataset.filter(self.dataset['track_uri'] == track_uri)['track_name'][0]
            for track_uri in self.track_vocab[:-1]
        ] + ["PAD"]

    def _make_dataset(self) -> pl.DataFrame:
        """
        Returns polars dataframe containing:
            pid: playlist id
            pos: position of the track in the playlist
            track_idx: internal index of the track
        """
        query = f"""
            SELECT PT.track_uri, PT.pid, PT.pos, T.track_name
            FROM playlist_track PT
            JOIN track T ON PT.track_uri = T.track_uri
            WHERE PT.pid IN (SELECT pid FROM playlist LIMIT {self.n_playlists});
        """
        # Playlist tracks as dataframe
        df_pl_tracks = pl.read_database(query, self.db.get_connection())

        # Get a list of unique tracks
        self.track_vocab = df_pl_tracks["track_uri"].unique().to_list() + ["PAD"]
        self.track_2_idx = {track: idx for idx, track in enumerate(self.track_vocab)}
        self.idx_2_track = lambda idx: self.track_vocab[idx]

        # Map track_uri to track_idx
        df_pl_tracks = df_pl_tracks.with_columns(
            track_idx=pl.col("track_uri").replace(
                self.track_2_idx,
                return_dtype=pl.UInt32
            )
        )
        return df_pl_tracks

    @property
    def n_tracks(self) -> int:
        return len(self.track_vocab)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Assumes idx is the index of the out-of-place track.
        Gets previous context_size tracks.
        """
        target = self.dataset["track_idx"][idx]

        # Get previous context_size tracks
        pid = self.dataset["pid"][idx]
        pos = self.dataset["pos"][idx]
        context = self.dataset.filter(
            (self.dataset["pid"] == pid) &
            (self.dataset["pos"] < pos)
        ).tail(self.context_size)["track_idx"].to_list()

        # Pad context if necessary
        if len(context) < self.context_size:
            pad_size = (self.context_size - len(context))
            context = [self.track_2_idx["PAD"]] * pad_size + context

        # Convert to pytorch tensors
        context = torch.tensor(context)
        # no need for one-hot encoding with NLLLoss
        return context, target
