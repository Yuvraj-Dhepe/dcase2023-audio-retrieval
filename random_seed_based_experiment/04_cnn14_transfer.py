import torch

# %%
from models import audio_encoders
from utils import audio_encoder_layer_map as ael

if __name__ == "__main__":
    # Call the transfer function
    ael.transfer_cnn_14_params(
        weights_path="./pretrained_models_weights/cnn14.pth",
        output_path="./pretrained_models_weights/CNN14_300.pth",
        layer_name_mapping=ael.cnn14_transfer_layer_mapping(),
    )
