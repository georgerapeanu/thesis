import unittest
from unittest import mock
import polars as pl

from data import extract_commentaries_for_svm


class TestExtractCommentariesForSVM(unittest.TestCase):
    @mock.patch('data.extract_commentaries_for_svm.pl.read_parquet')
    @mock.patch('data.extract_commentaries_for_svm.os.listdir', return_value=['1.parquet', '2.parquet', '3.parquet'])
    @mock.patch('data.extract_commentaries_for_svm.open')
    def test_extract(self, mock_open: mock.Mock, mock_listdir: mock.Mock, mock_read_parquet: mock.Mock) -> None:
        def read_parquet(x):
            if x == 'raw_data/train/1.parquet':
                return pl.DataFrame([
                    {
                        'previous_board': "",
                        'current_board': "",
                        "comment": "asfas"
                    },
                    {
                        'previous_board': "",
                        'current_board': "",
                        "comment": "as\nfas2"
                    },
                ])
            elif x == 'raw_data/train/2.parquet':
                return pl.DataFrame([
                    {
                        'previous_board': "",
                        'current_board': "",
                        "comment": ""
                    },
                    {
                        'previous_board': "",
                        'current_board': "",
                        "comment": "test3"
                    },
                ])
            elif x == 'raw_data/train/3.parquet':
                raise pl.exceptions.ComputeError()
            else:
                self.assertEqual(True, False, f"called with invalid {x}")

        mock_f = mock.Mock()
        mock_open.return_value.__enter__.return_value = mock_f
        mock_f.write()
        mock_read_parquet.side_effect = read_parquet

        written_lines = []

        def f_write(x):
            nonlocal written_lines
            written_lines.append(x)
            print(f"Debug {written_lines}, {x}")

        mock_f.write.side_effect = f_write

        extract_commentaries_for_svm.extract('artifacts', 'raw_data')
        mock_listdir.assert_called_with('raw_data/train')
        mock_open.assert_called_with('artifacts/commentaries_raw.txt', "w")

        self.assertListEqual(written_lines, ['asfas\n', 'as<n>fas2\n', 'test3\n'])

