import pickle
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from cbow_model.dataset import CBOWDataset
from data_storage.DBManager import DBManager


class Track2Vec(nn.Module):
    def __init__(self, num_tracks: int, embedding_dim: int, context_size: int):
        super(Track2Vec, self).__init__()
        self.embedding = nn.Embedding(num_tracks, embedding_dim)
        self.model = nn.Sequential(
            nn.Flatten(start_dim=1),  # concatenate context
            nn.Linear(context_size * embedding_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_tracks),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        embeddings = self.embedding(x)
        return self.model(embeddings)


def train(model: Track2Vec, ds_train: CBOWDataset, epochs: int, batch_size:int) -> Track2Vec:
    print("Training CBOW model")
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    data_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}")
        total_loss = 0
        for context, target in tqdm(data_loader):
            context = context.to(DEVICE)
            target = target.to(DEVICE)

            model.zero_grad()
            log_probs = model(context)
            loss = criterion(log_probs, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}: Loss: {total_loss}")
    return model


def save_model(model: Track2Vec, track2idx: dict, path: str):
    """
    Save the model state and the track2idx mapping to disk.
    We save the mapping because this is not captured in the model state,
    and changes depending on how many playlists are used to train the model.

    Args:
        model (Track2Vec): Trained model
        track2idx (dict): Mapping of track_uri to index
        path (str): Path to save the model
    """
    torch.save(model.state_dict(), path)

    # Change the extension to .pkl
    mapping_path = path.replace(".pt", f"-track2idx.pkl")
    with open(mapping_path, "wb") as f:
        pickle.dump(track2idx, f)


def load_model(
    model_path: str,
    mapping_path: str,
    context_size: int,
    embedding_dim: int,
) -> tuple[Track2Vec, dict]:
    """
    Load the model state and the track2idx mapping from disk.

    Args:
        model_path (str): Path to the model state
        mapping_path (str): Path to the track2idx mapping
        context_size (int): Context size used to train the model
        embedding_dim (int): Dimension of the embeddings

    Returns:
        tuple[Track2Vec, dict]: Model and track2idx mapping
    """
    model = Track2Vec(
        num_tracks=len(pickle.load(open(mapping_path, "rb"))),
        embedding_dim=embedding_dim,
        context_size=context_size,
    )
    model.load_state_dict(torch.load(model_path))
    with open(mapping_path, "rb") as f:
        track2idx = pickle.load(f)
    return model, track2idx


def to_tensorboard(model: Track2Vec, ds: CBOWDataset, run_name: str):
    writer = SummaryWriter(f"./runs/{run_name}/")
    writer.add_embedding(
        model.embedding.weight,
        metadata=ds.get_named_tracks(),
        tag=f"Track2Vec-CBOW",
    )
    writer.close()


if __name__ == "__main__":
    RUN_NAME = f"cbow_model@{time.strftime('%Y-%m-%d-%H-%M-%S')}"
    CONTEXT_SIZE = 5
    N_PLAYLISTS = 10000
    MIN_FREQ = 4
    EMBEDDING_DIM = 64
    N_EPOCHS = 1
    BATCH_SIZE = 32
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    db = DBManager()
    ds = CBOWDataset(db, n_playlists=N_PLAYLISTS, context_size=CONTEXT_SIZE, min_freq=MIN_FREQ)
    model = Track2Vec(
        num_tracks=ds.n_tracks, embedding_dim=EMBEDDING_DIM, context_size=CONTEXT_SIZE
    ).to(DEVICE)

    trained_model = train(
        model, ds, epochs=N_EPOCHS, batch_size=BATCH_SIZE
    )
    save_model(
        trained_model,
        ds.track_2_idx,
        f"./model_states/{RUN_NAME}_con{CONTEXT_SIZE}_pl{N_PLAYLISTS}_emb{EMBEDDING_DIM}_ep{N_EPOCHS}.pt",
    )

    to_tensorboard(
        trained_model, ds, run_name=RUN_NAME
    )

    db.disconnect()
