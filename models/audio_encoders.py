import torch
import torch.nn as nn

from models import init_weights


class CNN14Encoder(nn.Module):

    def __init__(self, *args, **kwargs):
        super(CNN14Encoder, self).__init__()
        n = 64
        self.bn0 = nn.BatchNorm2d(n * 1)

        self.cnn = nn.Sequential(
            # Conv2D block1
            nn.Conv2d(
                1, n * 1, kernel_size=3, stride=1, padding=1, bias=False
            ),  # 0
            nn.BatchNorm2d(n * 1),  # 1
            nn.ReLU(),  # 2
            nn.Conv2d(
                n * 1, n * 1, kernel_size=3, stride=1, padding=1, bias=False
            ),  # 3
            nn.BatchNorm2d(n * 1),  # 4
            nn.ReLU(),  # 5
            nn.AvgPool2d(kernel_size=2),  # 6
            nn.Dropout(p=kwargs["conv_dropout"]),  # 7
            # Conv2D block2
            nn.Conv2d(
                n * 1, n * 2, kernel_size=3, stride=1, padding=1, bias=False
            ),  # 8
            nn.BatchNorm2d(n * 2),  # 9
            nn.ReLU(),  # 10
            nn.Conv2d(
                n * 2, n * 2, kernel_size=3, stride=1, padding=1, bias=False
            ),  # 11
            nn.BatchNorm2d(n * 2),  # 12
            nn.ReLU(),  # 13
            nn.AvgPool2d(kernel_size=2),  # 14
            nn.Dropout(p=kwargs["conv_dropout"]),  # 15
            # Conv2D block3
            nn.Conv2d(
                n * 2, n * 4, kernel_size=3, stride=1, padding=1, bias=False
            ),  # 16
            nn.BatchNorm2d(n * 4),  # 17
            nn.ReLU(),  # 18
            nn.Conv2d(
                n * 4, n * 4, kernel_size=3, stride=1, padding=1, bias=False
            ),  # 19
            nn.BatchNorm2d(n * 4),  # 20
            nn.ReLU(),  # 21
            nn.AvgPool2d(kernel_size=2),  # 22
            nn.Dropout(p=kwargs["conv_dropout"]),  # 23
            # Conv2D block4
            nn.Conv2d(
                n * 4, n * 8, kernel_size=3, stride=1, padding=1, bias=False
            ),  # 24
            nn.BatchNorm2d(n * 8),  # 25
            nn.ReLU(),  # 26
            nn.Conv2d(
                n * 8, n * 8, kernel_size=3, stride=1, padding=1, bias=False
            ),  # 27
            nn.BatchNorm2d(n * 8),  # 28
            nn.ReLU(),  # 29
            nn.AvgPool2d(kernel_size=2),  # 30
            nn.Dropout(p=kwargs["conv_dropout"]),  # 31
            # Conv2D block5
            nn.Conv2d(
                n * 8, n * 16, kernel_size=3, stride=1, padding=1, bias=False
            ),  # 32
            nn.BatchNorm2d(n * 16),  # 33
            nn.ReLU(),  # 34
            nn.Conv2d(
                n * 16, n * 16, kernel_size=3, stride=1, padding=1, bias=False
            ),  # 35
            nn.BatchNorm2d(n * 16),  # 36
            nn.ReLU(),  # 37
            nn.AvgPool2d(kernel_size=2),  # 38
            nn.Dropout(p=kwargs["conv_dropout"]),  # 39
            # Conv2D block6
            nn.Conv2d(
                n * 16, n * 32, kernel_size=3, stride=1, padding=1, bias=False
            ),  # 40
            nn.BatchNorm2d(n * 32),  # 41
            nn.ReLU(),  # 42
            nn.Conv2d(
                n * 32, n * 32, kernel_size=3, stride=1, padding=1, bias=False
            ),  # 43
            nn.BatchNorm2d(n * 32),  # 44
            nn.ReLU(),  # 45
            nn.AvgPool2d(kernel_size=2),  # 46
            nn.Dropout(p=kwargs["conv_dropout"]),  # 47
        )

        self.fc = nn.Sequential(
            nn.Dropout(p=kwargs["fc_dropout"]),
            nn.Linear(n * 32, n * 32, bias=True),
            nn.ReLU(),
            nn.Dropout(p=kwargs["fc_dropout"]),
        )

        # NOTE: Original Baseline & {num}_x config
        # self.fc2 = nn.Linear(kwargs["fc_units"], kwargs["out_dim"], bias=True)

        # self.bn0.apply(init_weights)
        # self.cnn.apply(init_weights)
        # self.fc.apply(init_weights)
        # self.fc2.apply(init_weights)

        # NOTE: Finetuning baseline, using plane Linear
        self.fc1 = nn.Sequential(
            nn.Dropout(p=kwargs["fc_dropout"]),
            nn.Linear(n * 32, kwargs["fc_units"], bias=True),
            nn.ReLU(),
            nn.Dropout(p=kwargs["fc_dropout"]),
        )

        self.fc1 = nn.Linear(n * 32, kwargs["fc_units"], bias=True)
        self.fc2 = nn.Linear(kwargs["fc_units"], kwargs["out_dim"], bias=True)
        self.bn0.apply(init_weights)
        self.cnn.apply(init_weights)
        self.fc.apply(init_weights)
        self.fc1.apply(init_weights)
        self.fc2.apply(init_weights)

    def forward(self, x):
        """
        :param x: tensor, (batch_size, time_steps, Mel_bands).
        :return: tensor, (batch_size, embed_dim).
        """
        x = x.unsqueeze(1)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.cnn(x)
        x = torch.mean(x, dim=3)  # (N, n*32, T/n*1)

        (x1, _) = torch.max(x, dim=2)  # max across time
        x2 = torch.mean(x, dim=2)  # average over time
        x = x1 + x2  # (N, n*32)

        # NOTE: Original Baseline & {num}_x config
        # x = self.fc(x)  # (N, n*32)
        # x = self.fc2(x)  # (N, embed_dim)

        # NOTE: Finetuning baseline
        x = self.fc(x)  # (N, n*32)
        x = self.fc1(x)  # (N,fc_units)
        x = self.fc2(x)  # (N, embed_dim)

        return x
