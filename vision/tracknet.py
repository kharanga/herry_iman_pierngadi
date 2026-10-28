from pathlib import Path

import torch
from torch import nn


class Conv2DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same", bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(images)))


class Double2DConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_1 = Conv2DBlock(in_channels, out_channels)
        self.conv_2 = Conv2DBlock(out_channels, out_channels)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.conv_2(self.conv_1(images))


class Triple2DConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_1 = Conv2DBlock(in_channels, out_channels)
        self.conv_2 = Conv2DBlock(out_channels, out_channels)
        self.conv_3 = Conv2DBlock(out_channels, out_channels)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.conv_3(self.conv_2(self.conv_1(images)))


def halved(images: torch.Tensor) -> torch.Tensor:
    return nn.functional.max_pool2d(images, kernel_size=2, stride=2)


def doubled(images: torch.Tensor) -> torch.Tensor:
    return nn.functional.interpolate(images, scale_factor=2, mode="nearest")


class TrackNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down_block_1 = Double2DConv(in_channels, 64)
        self.down_block_2 = Double2DConv(64, 128)
        self.down_block_3 = Triple2DConv(128, 256)
        self.bottleneck = Triple2DConv(256, 512)
        self.up_block_1 = Triple2DConv(768, 256)
        self.up_block_2 = Double2DConv(384, 128)
        self.up_block_3 = Double2DConv(192, 64)
        self.predictor = nn.Conv2d(64, out_channels, (1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        full_scale = self.down_block_1(images)
        half_scale = self.down_block_2(halved(full_scale))
        quarter_scale = self.down_block_3(halved(half_scale))
        features = self.bottleneck(halved(quarter_scale))
        features = self.up_block_1(torch.cat([doubled(features), quarter_scale], dim=1))
        features = self.up_block_2(torch.cat([doubled(features), half_scale], dim=1))
        features = self.up_block_3(torch.cat([doubled(features), full_scale], dim=1))
        return self.sigmoid(self.predictor(features))


class Conv1DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding="same", bias=True)
        self.relu = nn.LeakyReLU()

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(sequence))


class Double1DConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_1 = Conv1DBlock(in_channels, out_channels)
        self.conv_2 = Conv1DBlock(out_channels, out_channels)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.conv_2(self.conv_1(sequence))


class InpaintNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_1 = Conv1DBlock(3, 32)
        self.down_2 = Conv1DBlock(32, 64)
        self.down_3 = Conv1DBlock(64, 128)
        self.buttleneck = Double1DConv(128, 256)
        self.up_1 = Conv1DBlock(384, 128)
        self.up_2 = Conv1DBlock(192, 64)
        self.up_3 = Conv1DBlock(96, 32)
        self.predictor = nn.Conv1d(32, 2, 3, padding="same")
        self.sigmoid = nn.Sigmoid()

    def forward(self, coordinates: torch.Tensor, inpaint_mask: torch.Tensor) -> torch.Tensor:
        sequence = torch.cat([coordinates, inpaint_mask], dim=2).permute(0, 2, 1)
        level_1 = self.down_1(sequence)
        level_2 = self.down_2(level_1)
        level_3 = self.down_3(level_2)
        features = self.buttleneck(level_3)
        features = self.up_1(torch.cat([features, level_3], dim=1))
        features = self.up_2(torch.cat([features, level_2], dim=1))
        features = self.up_3(torch.cat([features, level_1], dim=1))
        return self.sigmoid(self.predictor(features)).permute(0, 2, 1)


def require_checkpoint_file(checkpoint_path: Path) -> None:
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path} (run ./vision/fetch_tracknetv3.sh)")


def read_checkpoint(checkpoint_path: Path) -> dict:
    require_checkpoint_file(checkpoint_path)
    return torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)


def tracknet_input_channels(sequence_length: int, background_mode: str) -> int:
    if background_mode == "concat":
        return (sequence_length + 1) * 3
    raise ValueError(f"unsupported TrackNet background mode: {background_mode!r} (only 'concat' is implemented)")


def load_tracknet(checkpoint_path: Path) -> tuple[TrackNet, int]:
    checkpoint = read_checkpoint(checkpoint_path)
    sequence_length = int(checkpoint["param_dict"]["seq_len"])
    background_mode = checkpoint["param_dict"]["bg_mode"]
    network = TrackNet(tracknet_input_channels(sequence_length, background_mode), sequence_length)
    network.load_state_dict(checkpoint["model"])
    return network.eval(), sequence_length


def load_inpaintnet(checkpoint_path: Path) -> tuple[InpaintNet, int]:
    checkpoint = read_checkpoint(checkpoint_path)
    network = InpaintNet()
    network.load_state_dict(checkpoint["model"])
    return network.eval(), int(checkpoint["param_dict"]["seq_len"])
