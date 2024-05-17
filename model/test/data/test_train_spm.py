import unittest
from unittest import mock
import polars as pl

from data import spm_train


class TestExtractCommentariesForSVM(unittest.TestCase):
    @mock.patch('data.spm_train.spm.SentencePieceTrainer.train')
    def test_extract(self, mock_train: mock.Mock) -> None:
        def train_side_effect(*args, **kwargs):
            self.assertEqual(kwargs.get("vocab_size"), 20)
        mock_train.side_effect = train_side_effect
        spm_train.train_spm("_artifacts", 20)


