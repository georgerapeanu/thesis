import unittest
from unittest import mock
import polars as pl

from data import train_svm


class TestExtractCommentariesForSVM(unittest.TestCase):
    @mock.patch('data.train_svm.pl.read_parquet')
    @mock.patch('data.train_svm.pickle.dump')
    @mock.patch('data.train_svm.open')
    def test_extract(self, mock_open: mock.Mock, mock_pickle: mock.Mock, mock_read_parquet: mock.Mock) -> None:
        def read_parquet(x):
            if x == 'artifacts/commentary_types.parquet':
                return pl.DataFrame([
                    {
                        "commentary": "So he played it.",
                        "type": "4",
                        "type_str": "Context"
                    },
                    {
                        "commentary": "Maintaining the pin.",
                        "type": "0,3",
                        "type_str": "MoveDesc,Strategy"
                    },
                    {
                        "commentary": "I think h6 was a better move, opening a secure space for his bishop and defending g5, which my queen and knight can now freely use.",
                        "type": "2, 3",
                        "type_str": "Comparative, Strategy"
                    },
                    {
                        "commentary": "Tanto si las blancas toman de caballo como si toman de peón el repertorio se va a centrar en esta jugada",
                        "type": "5",
                        "type_str": "General"
                    },
                    {
                        "commentary": "I dont like that move. it seems very passive for white.",
                        "type": "1",
                        "type_str": "MoveQuality"
                    }
                ])
            else:
                self.assertEqual(True, False, f"called with invalid {x}")

        mock_f = mock.Mock()
        mock_open.return_value.__enter__.return_value = mock_f
        mock_f.write()
        mock_read_parquet.side_effect = read_parquet

        train_svm.train_svm("artifacts")
        mock_pickle.assert_called()
        mock_open.assert_called_with('artifacts/svm.p', "wb")

