import csv
import os
from collections import namedtuple
from typing import Any, Callable, List, Optional, Union, Tuple

import PIL
import torch

from torchvision.datasets.utils import (
    download_file_from_google_drive,
    verify_str_arg,
    extract_archive,
)
from torchvision.datasets.vision import VisionDataset

# Adapted from torchvision.datasets.CelebA

CSV = namedtuple("CSV", ["header", "index", "data"])


class CelebA(VisionDataset):
    base_folder = "celeba"
    file_list = [
        # File ID                                      MD5 Hash                            Filename
        (
            "0B7EVK8r0v71pZjFTYXZWM3FlRnM",
            "00d2c5bc6d35e252742224ab0c1e8fcb",
            "img_align_celeba.zip",
        ),
        (
            "0B7EVK8r0v71pY0NSMzRuSXJEVkk",
            "d32c9cbf5e040fd4025c592c306e6668",
            "list_eval_partition.txt",
        ),
    ]

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        download: bool = False,
        limit: Optional[int] = None,
    ) -> None:
        super().__init__(root, transform=transform)
        self.split = split

        if download:
            self.download()

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[
            verify_str_arg(split.lower(), "split", ("train", "valid", "test", "all"))
        ]
        splits = self._load_csv("list_eval_partition.txt")

        mask = slice(None) if split_ is None else (splits.data == split_).squeeze()

        if mask == slice(None):  # if split == "all"
            self.filename = splits.index
        else:
            self.filename = [
                splits.index[i] for i in torch.squeeze(torch.nonzero(mask))
            ]

        if limit is not None:
            self.filename = self.filename[:limit]

    def _load_csv(
        self,
        filename: str,
        header: Optional[int] = None,
    ) -> CSV:
        with open(os.path.join(self.root, self.base_folder, filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1:]
        else:
            headers = []

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return CSV(headers, indices, torch.tensor(data_int))

    def download(self) -> None:
        for file_id, md5, filename in self.file_list:
            download_file_from_google_drive(
                file_id, os.path.join(self.root, self.base_folder), filename, md5
            )

        extract_archive(
            os.path.join(self.root, self.base_folder, "img_align_celeba.zip")
        )

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        x = PIL.Image.open(
            os.path.join(
                self.root, self.base_folder, "img_align_celeba", self.filename[index]
            )
        )

        if self.transform is not None:
            x = self.transform(x)

        return x

    def __len__(self) -> int:
        return len(self.filename)
