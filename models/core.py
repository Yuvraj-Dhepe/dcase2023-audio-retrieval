import torch.nn as nn
import torch
import torch.nn.functional as F

from models import audio_encoders, text_encoders

bn_keymap = {
    # Map for bn0 block
    "bn_block": [
        "bn0.weight",
        "bn0.bias",
        "bn0.running_mean",
        "bn0.running_var",
    ],
}
conv_keymap = {
    # Maps for CNN blocks
    "conv_block1": [
        "cnn.0.weight",
        "cnn.1.weight",
        "cnn.1.bias",
        "cnn.3.weight",
        "cnn.4.weight",
        "cnn.4.bias",
    ],
    "conv_block2": [
        "cnn.8.weight",
        "cnn.9.weight",
        "cnn.9.bias",
        "cnn.11.weight",
        "cnn.12.weight",
        "cnn.12.bias",
    ],
    "conv_block3": [
        "cnn.16.weight",
        "cnn.17.weight",
        "cnn.17.bias",
        "cnn.19.weight",
        "cnn.20.weight",
        "cnn.20.bias",
    ],
    "conv_block4": [
        "cnn.24.weight",
        "cnn.25.weight",
        "cnn.25.bias",
        "cnn.27.weight",
        "cnn.28.weight",
        "cnn.28.bias",
    ],
    "conv_block5": [
        "cnn.32.weight",
        "cnn.33.weight",
        "cnn.33.bias",
        "cnn.35.weight",
        "cnn.36.weight",
        "cnn.36.bias",
    ],
    "conv_block6": [
        "cnn.40.weight",
        "cnn.41.weight",
        "cnn.41.bias",
        "cnn.43.weight",
        "cnn.44.weight",
        "cnn.44.bias",
    ],
}
fc_keymap = {
    # Map for fc block
    "fc_block": ["fc.1.weight", "fc.1.bias"],
}


class DualEncoderModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DualEncoderModel, self).__init__()

        self.out_norm = kwargs.get("out_norm", None)
        self.audio_enc = getattr(audio_encoders, args[0], None)(
            **kwargs["audio_enc"]
        )
        self.text_enc = getattr(text_encoders, args[1], None)(
            **kwargs["text_enc"]
        )

        # Load pretrained weights for audio encoder
        if kwargs["audio_enc"]["init"] == "prior":
            self.audio_enc.load_state_dict(kwargs["audio_enc"]["weight"])

            # Get trainable configuration from YAML
            conv_fine_tune_from = kwargs["audio_enc"].get(
                "conv_fine_tune_from", {}
            )

            set_requires_grad = False
            for name, param in self.audio_enc.named_parameters():
                # Always fine-tune BatchNorm and fully-connected layers
                if (
                    name in bn_keymap["bn_block"]
                    or name in fc_keymap["fc_block"]
                ):
                    param.requires_grad = True
                    continue

                # Fine-tune specific conv layers starting from `conv_fine_tune_from`
                for block, keys in conv_keymap.items():
                    if block == conv_fine_tune_from:
                        set_requires_grad = True  # Enable training for blocks starting from here

                    if set_requires_grad and name in keys:
                        param.requires_grad = True
                        break  # Stop checking conv blocks once match is found

                    # Freeze conv layers before `conv_fine_tune_from`
                    param.requires_grad = False

                # NOTE: The last fc layer i.e. fc2 from the architecture is a new addition which wasn't used in initializing pretrained weights so it is automatically fine_tuned

    def audio_branch(self, audio):
        audio_embeds = self.audio_enc(audio)

        if self.out_norm == "L2":
            audio_embeds = F.normalize(audio_embeds, p=2.0, dim=-1)

        return audio_embeds

    def text_branch(self, text):
        text_embeds = self.text_enc(text)

        if self.out_norm == "L2":
            text_embeds = F.normalize(text_embeds, p=2.0, dim=-1)

        return text_embeds

    def forward(self, audio, text):
        """
        :param audio: tensor, (batch_size, time_steps, Mel_bands).
        :param text: tensor, (batch_size, len_padded_text).
        """
        audio_embeds = self.audio_branch(audio)
        text_embeds = self.text_branch(text)

        # audio_embeds: [N, E]    text_embeds: [N, E]
        return audio_embeds, text_embeds
