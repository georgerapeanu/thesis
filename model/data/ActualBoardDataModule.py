import os

import lightning as L
import sentencepiece
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import crawler.main as crawler
import data.extract_commentaries_for_svm as extract_commentaries_for_svm
from data.ActualBoardCommentaryDataset import ActualBoardCommentaryDataset
from data.AlphazeroCommentaryDataset import AlphazeroCommentaryDataset
from data.extract_commentaries_for_spm import extract_spm
from data.process_raw_data import  process_raw_data
from data.spm_train import train_spm
from data.train_svm import train_svm
from omegaconf import DictConfig


class ActualBoardDataModule(L.LightningDataModule):
    def __init__(
        self,
        raw_data_path: str,
        processed_path: str,
        artifacts_path: str,
        pickle_path: str,
        engine_config: DictConfig,
        train_config: DictConfig,
        val_config: DictConfig,
        test_config: DictConfig,
        vocab_size: int,
        context_length: int,
        train_workers: int,
        test_workers: int,
        val_workers: int,
        force_recrawl: bool = False,
        force_reprocess: bool = False
    ):
        super().__init__()
        self.raw_data_path = raw_data_path
        self.processed_path = processed_path
        self.artifacts_path = artifacts_path
        self.pickle_path = pickle_path
        self.engine_config = engine_config
        self.train_config = train_config
        self.test_config = test_config
        self.val_config = val_config
        self.force_recrawl = force_recrawl
        self.force_reprocess = force_reprocess
        self.sp = None
        self.vocab_size = vocab_size

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_workers = train_workers
        self.val_workers = val_workers
        self.test_workers = test_workers

    def prepare_data(self) -> None:
        super().prepare_data()

        recrawl = self.force_recrawl
        for dir in ["train", "test", "valid"]:
            if not os.path.isdir(os.path.join(self.raw_data_path, dir)):
                os.mkdir(os.path.join(self.raw_data_path, dir))
                recrawl = True
        if recrawl:
            print("Crawling dataset")
            crawler.crawl(self.pickle_path, self.raw_data_path)

        reprocess = recrawl or self.force_reprocess
        for dir in ["train", "test", "valid"]:
            if not os.path.isdir(os.path.join(self.processed_path, dir)):
                os.mkdir(os.path.join(self.processed_path, dir))
                reprocess = True

        if reprocess:
            extract_commentaries_for_svm.extract(self.artifacts_path, self.raw_data_path)
            train_svm(self.artifacts_path)
            process_raw_data(self.engine_config)
            extract_spm(self.artifacts_path)

        if reprocess or not os.path.exists(os.path.join(self.artifacts_path, f"sp{self.vocab_size}.model")):
            print(f"Creating vocab{self.vocab_size}")
            train_spm(artifacts_path=self.artifacts_path, vocab_size=self.vocab_size)
        self.sp = sentencepiece.SentencePieceProcessor(os.path.join(self.artifacts_path, f"sp{self.vocab_size}.model"))

        self.force_reprocess = False
        self.force_recrawl = False

    def setup(self, stage: str) -> None:
        super().setup(stage)

        if stage == "fit":
            self.train_dataset = ActualBoardCommentaryDataset(
                self.train_config,
                self.engine_config,
                self.sp
            )
            self.val_dataset = ActualBoardCommentaryDataset(
                self.val_config,
                self.engine_config,
                self.sp
            )
        elif stage == "test":
            self.test_dataset = ActualBoardCommentaryDataset(
                self.test_config,
                self.engine_config,
                self.sp
            )
        else:
            raise Exception(f"Unknown stage: {stage}")

    def teardown(self, stage: str) -> None:
        if stage == "fit":
            del self.train_dataset
            del self.val_dataset
        elif stage == "test":
            del self.test_dataset
        else:
            raise Exception(f"Unknown stage: {stage}")

    # Batch x Board x 64, Batch x Board x 1 (strength) , Batch x Board x 1 (rep), Batch x 7(state features), tokens, types
    def get_collate_fn(self):
        def collate_fn(data):
            board_data = torch.stack(list(map(lambda x: x[0], data)))
            strength_data = torch.stack(list(map(lambda x: x[1], data)))
            rep_data = torch.stack(list(map(lambda x: x[2], data)))
            state_data = torch.stack(list(map(lambda x: x[3], data)))
            sequences = pad_sequence(list(map(lambda x: x[4], data)), batch_first=True,
                                     padding_value=self.sp.pad_id())
            types = torch.stack(list(map(lambda x: x[5], data)))
            X_sequence = sequences[:, :-1]
            y_sequence = sequences[:, 1:]
            next_pad_mask = (y_sequence == self.sp.pad_id())
            return board_data, strength_data, rep_data, state_data, X_sequence, y_sequence, next_pad_mask, types
        return collate_fn

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_config.batch_size,
            collate_fn=self.get_collate_fn(),
            num_workers=self.val_workers
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_config.batch_size,
            collate_fn=self.get_collate_fn(),
            num_workers=self.train_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_config.batch_size,
            collate_fn=self.get_collate_fn(),
            num_workers=self.test_workers
        )

    @staticmethod
    def get_board_token_size():
        return ActualBoardCommentaryDataset.get_board_token_size()