import torch
import torch.nn as nn
import torch.nn.functional as F


class LogSoftmaxLoss(nn.Module):

    def __init__(self, **kwargs):
        super(LogSoftmaxLoss, self).__init__()

        self.temperature = kwargs["temperature"]
        self.dist = kwargs.get("dist", "dot_product")

    def forward(self, audio_embeds, text_embeds, item_batch):
        """
        :param audio_embeds: tensor, (N, E).
        :param text_embeds: tensor, (N, E).
        :param item_batch: list of audio-text infos.
        :return:
        """
        N = audio_embeds.size(0)

        loss = torch.tensor(
            0.0, device=audio_embeds.device, requires_grad=True
        )

        for i in range(N):
            # Anchor audio-text pair
            A_i, T_i = audio_embeds[i], text_embeds[i]

            # Negative + Anchor audio-text pairs
            sample_indexes = [
                j
                for j in range(N)
                if item_batch[j]["fid"] != item_batch[i]["fid"]
            ]
            sample_indexes.append(i)

            # NOTE: Taking dot product of all the 64 audio embeds with a single text embed, where the last audio embed is the model pred for the T_i
            S_ai = (
                score(audio_embeds[sample_indexes], T_i, self.dist)
                / self.temperature
            )  # (N')

            # NOTE: Taking dot product of all the 64 text embeds with a single audio embed, where the last text embed is the model pred for the A_i
            S_it = (
                score(A_i, text_embeds[sample_indexes], self.dist)
                / self.temperature
            )  # (N')

            target = torch.as_tensor(
                [j == i for j in sample_indexes],
                dtype=torch.float,
                device=audio_embeds.device,
            )  # (N')

            # Log softmax loss (i.e., InfoNCE Loss, NT-Xent Loss, Multi-class N-pair Loss, Categorical CE Loss)
            L_ai = F.cross_entropy(S_ai, target)
            L_it = F.cross_entropy(S_it, target)

            loss = loss + L_ai + L_it

        loss = loss / N

        return loss


def score(audio_embed, text_embed, dist):
    """
    :param audio_embed: tensor, (E,) or (N, E).
    :param text_embed: tensor, (E,) or (N, E).
    """

    if dist == "dot_product":
        return torch.matmul(audio_embed, text_embed.t())

    elif dist == "cosine_similarity":
        return F.cosine_similarity(audio_embed, text_embed, -1, 1e-8)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        """
        Args:
            patience (int): How many epochs to wait before stopping when there's no improvement.
            min_delta (float): Minimum change to consider as an improvement.
            path (str): Path to save the model with best performance.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        # If best_loss is None, initialize with the first validation loss
        if self.best_loss is None:
            self.best_loss = val_loss

        elif val_loss < self.best_loss - self.min_delta:
            # Improvement found
            self.best_loss = val_loss
            self.counter = 0
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
