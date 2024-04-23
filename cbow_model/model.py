import os

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

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
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        embeddings = self.embedding(x)
        return self.model(embeddings)


def train(model: Track2Vec, ds_train: CBOWDataset, epochs : int) -> Track2Vec:
    print("Training CBOW model")
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    data_loader = DataLoader(ds_train, batch_size=32, shuffle=True)
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}")
        total_loss = 0
        for context, target in tqdm(data_loader):
            model.zero_grad()
            log_probs = model(context)
            loss = criterion(log_probs, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}: Loss: {total_loss}")
    return model


def to_tensorboard(model: Track2Vec, ds: CBOWDataset, run_name: str):
    writer = SummaryWriter(f'./runs/{run_name}/')
    writer.add_embedding(
        model.embedding.weight,
        metadata=ds.get_named_tracks(),
        tag=f"Track2Vec-CBOW",
    )
    writer.close()


if __name__ == '__main__':
    CONTEXT_SIZE = 5
    N_PLAYLISTS = 10
    EMBEDDING_DIM = 250
    N_EPOCHS = 10

    db = DBManager()
    ds = CBOWDataset(db, n_playlists=N_PLAYLISTS, context_size=CONTEXT_SIZE)
    model = Track2Vec(
        num_tracks=ds.n_tracks,
        embedding_dim=EMBEDDING_DIM,
        context_size=CONTEXT_SIZE
    )
    trained_model = train(model, ds, epochs=N_EPOCHS)
    torch.save(
        trained_model.state_dict(),
        f"cbow_model_{CONTEXT_SIZE}_{N_PLAYLISTS}_{EMBEDDING_DIM}.pt"
    )

    to_tensorboard(trained_model, ds, run_name="cbow_model")
