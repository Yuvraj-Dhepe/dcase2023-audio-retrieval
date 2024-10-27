import torch
import torch.nn as nn

from models import init_weights


class CNN14Encoder(nn.Module):

    def __init__(self, *args, **kwargs):
        super(CNN14Encoder, self).__init__()

        self.bn0 = nn.BatchNorm2d(64)

        self.cnn = nn.Sequential(
            # Conv2D block1
            nn.Conv2d(
                1, 64, kernel_size=3, stride=1, padding=1, bias=False
            ),  # 0
            nn.BatchNorm2d(64),  # 1
            nn.ReLU(),  # 2
            nn.Conv2d(
                64, 64, kernel_size=3, stride=1, padding=1, bias=False
            ),  # 3
            nn.BatchNorm2d(64),  # 4
            nn.ReLU(),  # 5
            nn.AvgPool2d(kernel_size=2),  # 6
            nn.Dropout(p=0.2),  # 7
            # Conv2D block2
            nn.Conv2d(
                64, 128, kernel_size=3, stride=1, padding=1, bias=False
            ),  # 8
            nn.BatchNorm2d(128),  # 9
            nn.ReLU(),  # 10
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1, padding=1, bias=False
            ),  # 11
            nn.BatchNorm2d(128),  # 12
            nn.ReLU(),  # 13
            nn.AvgPool2d(kernel_size=2),  # 14
            nn.Dropout(p=0.2),  # 15
            # Conv2D block3
            nn.Conv2d(
                128, 256, kernel_size=3, stride=1, padding=1, bias=False
            ),  # 16
            nn.BatchNorm2d(256),  # 17
            nn.ReLU(),  # 18
            nn.Conv2d(
                256, 256, kernel_size=3, stride=1, padding=1, bias=False
            ),  # 19
            nn.BatchNorm2d(256),  # 20
            nn.ReLU(),  # 21
            nn.AvgPool2d(kernel_size=2),  # 22
            nn.Dropout(p=0.2),  # 23
            # Conv2D block4
            nn.Conv2d(
                256, 512, kernel_size=3, stride=1, padding=1, bias=False
            ),  # 24
            nn.BatchNorm2d(512),  # 25
            nn.ReLU(),  # 26
            nn.Conv2d(
                512, 512, kernel_size=3, stride=1, padding=1, bias=False
            ),  # 27
            nn.BatchNorm2d(512),  # 28
            nn.ReLU(),  # 29
            nn.AvgPool2d(kernel_size=2),  # 30
            nn.Dropout(p=0.2),  # 31
            # Conv2D block5
            nn.Conv2d(
                512, 1024, kernel_size=3, stride=1, padding=1, bias=False
            ),  # 32
            nn.BatchNorm2d(1024),  # 33
            nn.ReLU(),  # 34
            nn.Conv2d(
                1024, 1024, kernel_size=3, stride=1, padding=1, bias=False
            ),  # 35
            nn.BatchNorm2d(1024),  # 36
            nn.ReLU(),  # 37
            nn.AvgPool2d(kernel_size=2),  # 38
            nn.Dropout(p=0.2),  # 39
            # Conv2D block6
            nn.Conv2d(
                1024, 2048, kernel_size=3, stride=1, padding=1, bias=False
            ),  # 40
            nn.BatchNorm2d(2048),  # 41
            nn.ReLU(),  # 42
            nn.Conv2d(
                2048, 2048, kernel_size=3, stride=1, padding=1, bias=False
            ),  # 43
            nn.BatchNorm2d(2048),  # 44
            nn.ReLU(),  # 45
            nn.AvgPool2d(kernel_size=2),  # 46
            nn.Dropout(p=0.2),  # 47
        )

        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

        self.fc2 = nn.Linear(2048, kwargs["out_dim"], bias=True)

        self.bn0.apply(init_weights)
        self.cnn.apply(init_weights)
        self.fc.apply(init_weights)
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
        x = torch.mean(x, dim=3)  # (N, 2048, T/64)

        (x1, _) = torch.max(x, dim=2)  # max across time
        x2 = torch.mean(x, dim=2)  # average over time
        x = x1 + x2  # (N, 2048)

        x = self.fc(x)  # (N, 2048)
        x = self.fc2(x)  # (N, embed_dim)

        return x
