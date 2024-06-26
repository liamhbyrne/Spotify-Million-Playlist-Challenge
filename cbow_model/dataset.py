import logging

import numpy as np
import polars as pl
import torch
from torch import Tensor
from torch.utils.data import Dataset
from typing import List, Tuple

from tqdm import tqdm

from data_storage.DBManager import DBManager


class CBOWDataset(Dataset):
    """
    Selects tracks from the dataset on the SQLite database.
    """

    def __init__(
        self,
        db_manager: DBManager,
        n_playlists: int,
        context_size: int = 5,
        min_freq: int = 5,
        preloaded_dataset_path: str = None,
    ):
        self.n_playlists = n_playlists
        self.context_size = context_size
        self.min_freq = min_freq
        self.db = db_manager
        self.preloaded_dataset_path = preloaded_dataset_path
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        self.cur = self.db.get_cursor()
        self.dataset = self._make_dataset()
        self.cbow_dataset = self._preprocess()

    def get_named_tracks(self) -> List[str]:
        """
        Query DB to obtain artist names for each track, concatenate with track names.
        This is particularly useful for TensorBoard projections.
        """
        query = f"""
                    SELECT artist.artist_name, track.track_name
                    FROM track
                    JOIN album ON track.album_uri = album.album_uri
                    JOIN artist ON album.artist_uri = artist.artist_uri
                    WHERE track_uri IN ({','.join(['?']*(len(self.track_vocab)-1))});
                """

        artist_tracks = pl.read_database(
            query,
            self.db.get_connection(),
            execute_options={"parameters": self.track_vocab[:-1]},
        )

        named_tracks = artist_tracks["artist_name"] + ": " + artist_tracks["track_name"]
        named_tracks = named_tracks.to_list()
        return named_tracks + ["PAD"]

    def _make_dataset(self) -> pl.DataFrame:
        """
        Returns polars dataframe containing:
            pid: playlist id
            pos: position of the track in the playlist
            track_idx: internal index of the track
        """
        # Load dataset from pkl if available
        if self.preloaded_dataset_path:
            self.logger.info(f"Loading dataset from {self.preloaded_dataset_path}")

            df_pl_tracks = pl.read_csv(self.preloaded_dataset_path)

            self.track_vocab = df_pl_tracks["track_uri"].unique().to_list() + ["PAD"]
            self.track_2_idx = {
                track: idx for idx, track in enumerate(self.track_vocab)
            }
            self.idx_2_track = lambda idx: self.track_vocab[idx]

            return df_pl_tracks

        query = f"""
            SELECT PT.pid, PT.track_uri, PT.pos
            FROM playlist_track PT
            INNER JOIN (
                SELECT track_uri
                FROM playlist_track
                GROUP BY track_uri
                HAVING COUNT(pid) > {self.min_freq}
            ) AS PT2 ON PT.track_uri = PT2.track_uri
            WHERE pid < {self.n_playlists}
        """
        self.logger.info("Querying database for playlist tracks . . .")

        # Playlist tracks as dataframe
        df_pl_tracks = pl.read_database(query, self.db.get_connection())

        # Get a list of unique tracks
        self.track_vocab = df_pl_tracks["track_uri"].unique().to_list() + ["PAD"]
        self.track_2_idx = {track: idx for idx, track in enumerate(self.track_vocab)}
        self.idx_2_track = lambda idx: self.track_vocab[idx]

        # Map track_uri to track_idx
        df_pl_tracks = df_pl_tracks.with_columns(
            track_idx=pl.col("track_uri").replace(
                self.track_2_idx, return_dtype=pl.UInt32
            )
        )
        self.logger.info("Done.")

        return df_pl_tracks

    def save_dataset(self, path: str):
        assert path.endswith(".csv"), "Path must end with .csv"
        self.dataset.write_csv(path)

    def _preprocess(self) -> List[Tuple[Tensor, int]]:
        cbows = []
        self.logger.info("Preprocessing dataset . . .")
        for pid, tracks in tqdm(self.dataset.group_by(["pid"]), total=self.n_playlists):
            track_ids = np.concatenate([
                np.array([self.track_2_idx["PAD"]] * (self.context_size)),
                tracks["track_idx"].to_numpy()
            ])
            window_size = self.context_size + 1  # include target in window
            shape = (track_ids.size - window_size + 1, window_size)
            strides = (track_ids.itemsize, track_ids.itemsize)
            windows = np.lib.stride_tricks.as_strided(track_ids, shape=shape, strides=strides)
            for window in windows:
                cbows.append((torch.tensor(window[:-1]), window[-1]))
        return cbows

    @property
    def n_tracks(self) -> int:
        return len(self.track_vocab)

    def __len__(self):
        return len(self.cbow_dataset)

    def __getitem__(self, idx):
        """
        Assumes idx is the index of the out-of-place track.
        Gets previous context_size tracks.
        """
        return self.cbow_dataset[idx]
        
        # target = self.dataset["track_idx"][idx]
        # # target_ohe = torch.zeros(self.n_tracks)
        # # target_ohe[target] = 1

        # # Get previous context_size tracks
        # pid = self.dataset["pid"][idx]
        # pos = self.dataset["pos"][idx]
        # context = (
        #     self.dataset.filter(
        #         (self.dataset["pid"] == pid) & (self.dataset["pos"] < pos)
        #     )
        #     .tail(self.context_size)["track_idx"]
        #     .to_list()
        # )

        # # Pad context if necessary
        # if len(context) < self.context_size:
        #     pad_size = self.context_size - len(context)
        #     context = [self.track_2_idx["PAD"]] * pad_size + context

        # # Convert to pytorch tensors
        # context = torch.tensor(context)
        # # no need for one-hot encoding with NLLLoss
        # return context, target
