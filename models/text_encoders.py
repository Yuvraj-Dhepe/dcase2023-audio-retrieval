import torch
import torch.nn as nn

from models import init_weights


class SentBERTBaseEncoder(nn.Module):

    def __init__(self, *args, **kwargs):
        super(SentBERTBaseEncoder, self).__init__()

        # Embeddings
        self.embedding = nn.Embedding(
            num_embeddings=kwargs["num_embed"],
            embedding_dim=768,
            _weight=kwargs["weight"],
        )

        # Freeze embeddings
        for param in self.embedding.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(768, 512, bias=True)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(512, kwargs["out_dim"], bias=True)
        self.fc1.apply(init_weights)
        self.fc2.apply(init_weights)

    def forward(self, x):
        """
        :param x: tensor, (batch_size, len_padded_text).
        :return: tensor, (batch_size, embed_dim).
        """
        x = self.embedding(x)  # (batch_size, len_padded_text, embed_dim)

        x = torch.mean(x, dim=1, keepdim=False)

        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
