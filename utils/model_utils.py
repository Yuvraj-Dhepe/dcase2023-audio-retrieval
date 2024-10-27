import copy
import os
from tqdm import tqdm
import numpy as np
import torch

from models import core


def init_model(params, vocab):
    params = copy.deepcopy(params)

    # Text encoder params
    params["text_enc"]["num_embed"] = len(vocab)

    if params["text_enc"]["init"] == "prior":
        # Tid 2 Text Embedding
        id2vec = {
            vocab.key2id[key]: vocab.key2vec[key] for key in vocab.key2id
        }
        weights = np.array([id2vec[id] for id in range(vocab.id)])
        weights = torch.as_tensor(weights, dtype=torch.float)
        params["text_enc"]["weight"] = weights
    else:
        params["text_enc"]["weight"] = None

    # Audio encoder params
    if params["audio_enc"]["init"] == "prior":
        params["audio_enc"]["weight"] = torch.load(
            params["audio_enc"]["weight"], weights_only=True
        )
    else:
        params["audio_enc"]["weight"] = None

    # Initialize dual encoders
    model = getattr(core, "DualEncoderModel", None)(
        params["audio_enc"]["name"], params["text_enc"]["name"], **params
    )

    return model


# NOTE: Normal Train & Eval
# def train(model, data_loader, criterion, optimizer):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     criterion.to(device=device)
#     model.to(device=device)

#     model.train()

#     for batch_idx, data in enumerate(data_loader, 0):
#         item_batch, audio_batch, text_batch = data

#         audio_batch = audio_batch.to(device)
#         text_batch = text_batch.to(device)

#         # Zero the parameter gradients
#         optimizer.zero_grad()

#         # Forward + backward + optimize
#         audio_embeds, text_embeds = model(audio_batch, text_batch)
#         loss = criterion(audio_embeds, text_embeds, item_batch)
#         loss.backward()
#         optimizer.step()

# def eval(model, data_loader, criterion):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     criterion.to(device=device)
#     model.to(device=device)

#     model.eval()

#     eval_loss, eval_steps = 0.0, 0

#     with torch.no_grad():
#         for batch_idx, data in enumerate(data_loader, 0):
#             item_batch, audio_batch, text_batch = data

#             audio_batch = audio_batch.to(device)
#             text_batch = text_batch.to(device)

#             audio_embeds, text_embeds = model(audio_batch, text_batch)
#             loss = criterion(audio_embeds, text_embeds, item_batch)

#             eval_loss += loss.cpu().numpy()
#             eval_steps += 1

#     return eval_loss / max(eval_steps, 1)


# NOTE: TQDM Train & Eval
def tqdm_train(model, data_loader, criterion, optimizer, epoch, total_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion.to(device=device)
    model.to(device=device)

    model.train()

    # Wrap the data_loader with tqdm
    with tqdm(
        data_loader,
        unit="batch",
        desc=f"Training Epoch {epoch}/{total_epochs}",
    ) as tepoch:
        for batch_idx, data in enumerate(tepoch, 0):
            item_batch, audio_batch, text_batch = data

            audio_batch = audio_batch.to(device)
            text_batch = text_batch.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            audio_embeds, text_embeds = model(audio_batch, text_batch)
            loss = criterion(audio_embeds, text_embeds, item_batch)
            loss.backward()
            optimizer.step()

            # Update tqdm progress bar
            tepoch.set_postfix(loss=loss.item())


def eval(model, data_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion.to(device=device)
    model.to(device=device)

    model.eval()

    eval_loss, eval_steps = 0.0, 0

    with torch.no_grad():
        # Wrap the data_loader with tqdm
        with tqdm(data_loader, unit="batch", desc="Evaluating") as tepoch:
            for batch_idx, data in enumerate(tepoch, 0):
                item_batch, audio_batch, text_batch = data

                audio_batch = audio_batch.to(device)
                text_batch = text_batch.to(device)

                audio_embeds, text_embeds = model(audio_batch, text_batch)
                loss = criterion(audio_embeds, text_embeds, item_batch)

                eval_loss += loss.cpu().numpy()
                eval_steps += 1

                # Update tqdm progress bar (optional)
                tepoch.set_postfix(loss=loss.item())

    return eval_loss / max(eval_steps, 1)


def restore(model, ckp_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_state = torch.load(
        os.path.join(ckp_dir), map_location=device, weights_only=True
    )
    model.load_state_dict(model_state["model_state_dict"])
    return model
