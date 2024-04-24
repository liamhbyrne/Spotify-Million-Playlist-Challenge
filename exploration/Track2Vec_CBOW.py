"""
PoC for training track embeddings using a Continuous Bag of Words model with negative sampling over
spotify playlists.
"""

import logging
import pickle
import time

import pandas as pd
import torch
from torch import nn, Tensor, LongTensor
import torch.nn.functional as F
from typing import List, Tuple

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_storage.DBManager import DBManager

logging.basicConfig(level=logging.INFO)


class Track2VecTrainer:
    def __init__(
        self, embedding_dim: int, context_size: int, causal_cbow: bool
    ) -> None:
        self._db: DBManager = DBManager()
        self._embedding_dim = embedding_dim
        self._context_size = context_size
        self._causal_cbow = causal_cbow
        self._logger = logging.getLogger(self.__class__.__name__)

    def build_CBOW_dataset(
        self, n_playlists: int = 1000
    ) -> List[Tuple[List[int], int]]:
        """
        Build the dataset for the CBOW model.
        """

        self._logger.info("Building dataset for CBOW model")

        query = f"""
            SELECT PT.track_uri, PT.pid, PT.pos, track.track_name
            FROM playlist_track PT
            JOIN track ON PT.track_uri = track.track_uri
            -- Limit to N_PLAYLISTS
            WHERE PT.pid IN (SELECT pid FROM playlist LIMIT {n_playlists});
        """

        db_results = pd.read_sql(query, self._db.get_connection())

        # Get a list of unique tracks
        self._track_vocab = list(set(db_results["track_uri"].values)) + ["PAD"]

        self._logger.info(f"Number of unique tracks: {len(self._track_vocab)}")
        self._logger.info(
            f"Average number of tracks per playlist: {len(db_results) / n_playlists}"
        )

        # Create a mapping from track_uri to index
        self._track2idx = {track: idx for idx, track in enumerate(self._track_vocab)}
        self._idx2track = {idx: track for idx, track in enumerate(self._track_vocab)}

        # For each playlist, group the tracks by the playlist id and sort by position then call collate_causal_CBOW
        if self._causal_cbow:
            collate_fn = self.collate_causal_CBOW
        else:
            raise NotImplementedError

        training_data = []

        for pid, group in db_results.groupby("pid"):
            # Sort the tracks by position in the playlist
            tracks = group.sort_values("pos")["track_uri"].to_list()
            training_data.extend(collate_fn(tracks))

        self._logger.info(
            f"Dataset built. Number of training samples: {len(training_data)}"
        )

        return training_data

    def collate_causal_CBOW(self, tracks: List[str]) -> list[tuple[Tensor, LongTensor]]:
        """
        Collate the tracks into context and target pairs.
        For causal CBOW, the context is the tracks before the target track.

        Args:
            tracks (List[str]): List of track_uris

        Returns:
            List[Tuple[List[int], int]]: List of context and target pairs
        """
        pairs = []

        # Create a sliding window of size context_size
        for i, target in enumerate(tracks):
            context = tracks[max(0, i - self._context_size) : i]

            while len(context) < self._context_size:
                context.append("PAD")

            assert len(context) == self._context_size

            # Convert the context and target to indices
            context_idxs = [self._track2idx[track] for track in context]
            target_idx = self._track2idx[target]

            pairs.append((context_idxs, target_idx))

        return pairs

    def train(self, training_data: List[Tuple[List[int], int]], epochs: int) -> None:
        """
        Train the CBOW model.

        Args:
            training_data (List[Tuple[List[int], int]]): List of context and target pairs
        """
        self._logger.info("Training CBOW model")

        # Instantiate the model
        self._model = Track2Vec(
            num_tracks=len(self._track2idx),
            embedding_dim=self._embedding_dim,
            context_size=self._context_size,
        )

        # Loss function and optimizer
        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=0.001)

        for epoch in range(1, epochs + 1):
            self._logger.info(f"Epoch {epoch}")
            total_loss = 0
            for context, target in tqdm(training_data):
                context = torch.tensor(context, dtype=torch.long)
                target = torch.tensor([target], dtype=torch.long)

                # Concatenate
                self._model.zero_grad()

                log_probs = self._model(context)
                loss = criterion(log_probs, target)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch}: Loss: {total_loss}")

    def to_tensorboard(self, run_name: str):
        """
        Log the training metrics to tensorboard.
        """
        # Map self._track_vocab to the Artist and Track name using DB
        query = f"""
            SELECT artist.artist_name, track.track_name
            FROM track
            JOIN album ON track.album_uri = album.album_uri
            JOIN artist ON album.artist_uri = artist.artist_uri
            WHERE track_uri IN ({','.join(['?']*len(self._track_vocab))});
        """

        db_results = pd.read_sql(
            query, self._db.get_connection(), params=self._track_vocab
        )
        named_tracks = db_results.apply(
            lambda x: f"{x['artist_name']}: {x['track_name']}", axis=1
        ).values.tolist()

        named_tracks.append("PAD")

        writer = SummaryWriter(f"../tensorboard_runs/{run_name}")
        writer.add_embedding(
            self._model.embedding.weight,
            metadata=named_tracks,
            tag="Track2Vec-CBOW",
        )
        writer.close()

    @staticmethod
    def load_model(
        model_path: str,
        mapping_path: str,
        vocab_size: int,
        embedding_dim: int,
        context_length: int,
    ):

        model = Track2Vec(vocab_size, embedding_dim, context_length)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

        # unpickle the tag_to_ix
        with open(mapping_path, "rb") as f:
            model.tag_to_ix = pickle.load(f)

        return model


class Track2Vec(nn.Module):
    def __init__(self, num_tracks: int, embedding_dim: int, context_size: int):
        super(Track2Vec, self).__init__()
        self.embedding = nn.Embedding(num_tracks, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, num_tracks)

    def forward(self, x):
        embeds = self.embedding(x).view((1, -1))

        out = F.leaky_relu(self.linear1(embeds))
        out = self.linear2(out)

        log_probs = F.log_softmax(out, dim=1)
        return log_probs


if __name__ == "__main__":
    trainer = Track2VecTrainer(embedding_dim=32, context_size=5, causal_cbow=True)

    dataset = trainer.build_CBOW_dataset(n_playlists=1000)

    trainer.train(dataset, epochs=1)

    trainer.save_model(
        f"/scratch/dlt/models/track2vec@{time.strftime('%Y%m%d-%H%M%S')}.pt"
    )

    # Log to tensorboard use current datetime as run name
    trainer.to_tensorboard(f"/scratch/dlt/run@{time.strftime('%Y%m%d-%H%M%S')}")
