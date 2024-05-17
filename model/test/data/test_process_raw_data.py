import unittest
from unittest import mock
import polars as pl

from data import process_raw_data


class TestExtractCommentariesForSVM(unittest.TestCase):

    @mock.patch('data.process_raw_data.pickle.load')
    @mock.patch('data.process_raw_data.stockfish.Stockfish')
    @mock.patch('data.process_raw_data.os.listdir')
    @mock.patch('crawler.main.pl.DataFrame.write_parquet', autospec=True)
    @mock.patch('data.process_raw_data.multiprocessing.get_context')
    @mock.patch('data.process_raw_data.pl.read_parquet')
    @mock.patch('data.process_raw_data.open')
    def test_extract(self,
                     mock_open: mock.Mock,
                     mock_read_parquet: mock.Mock,
                     mock_pool_context: mock.Mock,
                     mock_write_parquet: mock.Mock,
                     mock_listdir: mock.Mock,
                     mock_stockfish: mock.Mock,
                     mock_pickle_load: mock.Mock
    ) -> None:
        mock_pool = mock.Mock()
        mock_pool_context.return_value.Pool.return_value.__enter__.return_value = mock_pool

        def mock_pool_map(f, args):
            answer = []
            for arg in args:
                answer.append(f(arg))
            return answer

        mock_pool.map.side_effect = mock_pool_map

        def listdir(path):
            if path == "raw_data_mock/train":
                return ["train.parquet", "train_empty.parquet"]
            elif path == "raw_data_mock/valid":
                return ["valid.parquet"]
            elif path == "raw_data_mock/test":
                return ["test.parquet"]
            else:
                self.assertEqual(True, False, f"called with invalid {path}")

        mock_listdir.side_effect = listdir
        mock_stockfish.return_value.get_evaluation.return_value = {
            'type': 'cp',
            'value': 10
        }

        (vectorizer_mock, classifiers_mock) = (mock.Mock(), [mock.MagicMock()])
        classifiers_mock[0].transform.return_value = [0, 1]

        mock_pickle_load.return_value = (vectorizer_mock, classifiers_mock)

        def read_parquet(path):
            if path == "raw_data_mock/train/train.parquet":
                return pl.DataFrame([
                    {
                        "previous_board": "",
                        "current_board": "4",
                        "commentary": "training"
                    },
                    {
                        "previous_board": "",
                        "current_board": "4",
                        "commentary": ""
                    }
                ])
            elif path == "raw_data_mock/train/train_empty.parquet":
                raise pl.exceptions.ComputeError()
            elif path == "raw_data_mock/valid/valid.parquet":
                return pl.DataFrame([
                    {
                        "previous_board": "",
                        "current_board": "4",
                        "commentary": "validing"
                    },
                    {
                        "previous_board": "",
                        "current_board": "4",
                        "commentary": ""
                    }
                ])
            elif path == "raw_data_mock/test/test.parquet":
                return pl.DataFrame([
                    {
                        "previous_board": "",
                        "current_board": "4",
                        "commentary": "testing"
                    },
                    {
                        "previous_board": "",
                        "current_board": "4",
                        "commentary": ""
                    }
                ])
            else:
                self.assertEqual(True, False, f"called with invalid {path}")
        mock_read_parquet.side_effect = read_parquet

        config = {
           'artifacts_path': "artifacts_mock",
            'location': "stockfish_mock",
            'threads': 4,
            'hash': 128,
            'minimum_thinking_time': 1,
            'raw_data_path': "raw_data_mock",
            'processed_path': "processed_data_mock",
            'engine_depth': 5,
            'mate_value': 10000
        }
        process_raw_data.init_worker(config)

        dfs_rows = []
        def mock_write_parquet_side_effect(df, file):
            nonlocal dfs_rows
            dfs_rows.append(df.rows())

        mock_write_parquet.side_effect = mock_write_parquet_side_effect

        process_raw_data.process_raw_data(config)

        mock_open.assert_called_with("artifacts_mock/svm.p", "rb")
        self.assertEqual(dfs_rows, [[('', '4', 'training', 10, 10, 1)], [('', '4', 'testing', 10, 10, 1)], [('', '4', 'validing', 10, 10, 1)]])

    @mock.patch('data.process_raw_data.pickle.load')
    @mock.patch('data.process_raw_data.stockfish.Stockfish')
    @mock.patch('data.process_raw_data.os.listdir')
    @mock.patch('crawler.main.pl.DataFrame.write_parquet', autospec=True)
    @mock.patch('data.process_raw_data.multiprocessing.get_context')
    @mock.patch('data.process_raw_data.pl.read_parquet')
    @mock.patch('data.process_raw_data.open')
    def test_extract_mate_eval(self,
                     mock_open: mock.Mock,
                     mock_read_parquet: mock.Mock,
                     mock_pool_context: mock.Mock,
                     mock_write_parquet: mock.Mock,
                     mock_listdir: mock.Mock,
                     mock_stockfish: mock.Mock,
                     mock_pickle_load: mock.Mock
                     ) -> None:
        mock_pool = mock.Mock()
        mock_pool_context.return_value.Pool.return_value.__enter__.return_value = mock_pool

        def mock_pool_map(f, args):
            answer = []
            for arg in args:
                answer.append(f(arg))
            return answer

        mock_pool.map.side_effect = mock_pool_map

        def listdir(path):
            if path == "raw_data_mock/train":
                return ["train.parquet"]
            elif path == "raw_data_mock/valid":
                return ["valid.parquet"]
            elif path == "raw_data_mock/test":
                return ["test.parquet"]
            else:
                self.assertEqual(True, False, f"called with invalid {path}")

        mock_listdir.side_effect = listdir
        mock_stockfish.return_value.get_evaluation.return_value = {
            'type': 'mate',
            'value': 10
        }

        (vectorizer_mock, classifiers_mock) = (mock.Mock(), [mock.MagicMock()])
        classifiers_mock[0].transform.return_value = [0, 1]

        mock_pickle_load.return_value = (vectorizer_mock, classifiers_mock)

        def read_parquet(path):
            if path == "raw_data_mock/train/train.parquet":
                return pl.DataFrame([
                    {
                        "previous_board": "",
                        "current_board": "4",
                        "commentary": "training"
                    },
                    {
                        "previous_board": "",
                        "current_board": "4",
                        "commentary": ""
                    }
                ])
            elif path == "raw_data_mock/train/train_empty.parquet":
                raise pl.exceptions.ComputeError()
            elif path == "raw_data_mock/valid/valid.parquet":
                return pl.DataFrame([
                    {
                        "previous_board": "",
                        "current_board": "4",
                        "commentary": "validing"
                    },
                    {
                        "previous_board": "",
                        "current_board": "4",
                        "commentary": ""
                    }
                ])
            elif path == "raw_data_mock/test/test.parquet":
                return pl.DataFrame([
                    {
                        "previous_board": "",
                        "current_board": "4",
                        "commentary": "testing"
                    },
                    {
                        "previous_board": "",
                        "current_board": "4",
                        "commentary": ""
                    }
                ])
            else:
                self.assertEqual(True, False, f"called with invalid {path}")
        mock_read_parquet.side_effect = read_parquet

        config = {
            'artifacts_path': "artifacts_mock",
            'location': "stockfish_mock",
            'threads': 4,
            'hash': 128,
            'minimum_thinking_time': 1,
            'raw_data_path': "raw_data_mock",
            'processed_path': "processed_data_mock",
            'engine_depth': 5,
            'mate_value': 10000
        }
        process_raw_data.init_worker(config)

        dfs_rows = []

        def mock_write_parquet_side_effect(df, file):
            nonlocal dfs_rows
            dfs_rows.append(df.rows())

        mock_write_parquet.side_effect = mock_write_parquet_side_effect

        process_raw_data.process_raw_data(config)

        mock_open.assert_called_with("artifacts_mock/svm.p", "rb")
        self.assertEqual(dfs_rows, [[('', '4', 'training', 10000, 10000, 1)], [('', '4', 'testing', 10000, 10000, 1)],
                                    [('', '4', 'validing', 10000, 10000, 1)]])

    @mock.patch('data.process_raw_data.pickle.load')
    @mock.patch('data.process_raw_data.stockfish.Stockfish')
    @mock.patch('data.process_raw_data.os.listdir')
    @mock.patch('crawler.main.pl.DataFrame.write_parquet', autospec=True)
    @mock.patch('data.process_raw_data.multiprocessing.get_context')
    @mock.patch('data.process_raw_data.pl.read_parquet')
    @mock.patch('data.process_raw_data.open')
    def test_extract_mate_eval_black(self,
                     mock_open: mock.Mock,
                     mock_read_parquet: mock.Mock,
                     mock_pool_context: mock.Mock,
                     mock_write_parquet: mock.Mock,
                     mock_listdir: mock.Mock,
                     mock_stockfish: mock.Mock,
                     mock_pickle_load: mock.Mock
                     ) -> None:
        mock_pool = mock.Mock()
        mock_pool_context.return_value.Pool.return_value.__enter__.return_value = mock_pool

        def mock_pool_map(f, args):
            answer = []
            for arg in args:
                answer.append(f(arg))
            return answer

        mock_pool.map.side_effect = mock_pool_map

        def listdir(path):
            if path == "raw_data_mock/train":
                return ["train.parquet"]
            elif path == "raw_data_mock/valid":
                return ["valid.parquet"]
            elif path == "raw_data_mock/test":
                return ["test.parquet"]
            else:
                self.assertEqual(True, False, f"called with invalid {path}")

        mock_listdir.side_effect = listdir
        mock_stockfish.return_value.get_evaluation.return_value = {
            'type': 'mate',
            'value': -3
        }

        (vectorizer_mock, classifiers_mock) = (mock.Mock(), [mock.MagicMock()])
        classifiers_mock[0].transform.return_value = [0, 1]

        mock_pickle_load.return_value = (vectorizer_mock, classifiers_mock)

        def read_parquet(path):
            if path == "raw_data_mock/train/train.parquet":
                return pl.DataFrame([
                    {
                        "previous_board": "",
                        "current_board": "4",
                        "commentary": "training"
                    },
                    {
                        "previous_board": "",
                        "current_board": "4",
                        "commentary": ""
                    }
                ])
            elif path == "raw_data_mock/train/train_empty.parquet":
                raise pl.exceptions.ComputeError()
            elif path == "raw_data_mock/valid/valid.parquet":
                return pl.DataFrame([
                    {
                        "previous_board": "",
                        "current_board": "4",
                        "commentary": "validing"
                    },
                    {
                        "previous_board": "",
                        "current_board": "4",
                        "commentary": ""
                    }
                ])
            elif path == "raw_data_mock/test/test.parquet":
                return pl.DataFrame([
                    {
                        "previous_board": "",
                        "current_board": "4",
                        "commentary": "testing"
                    },
                    {
                        "previous_board": "",
                        "current_board": "4",
                        "commentary": ""
                    }
                ])
            else:
                self.assertEqual(True, False, f"called with invalid {path}")
        mock_read_parquet.side_effect = read_parquet

        config = {
            'artifacts_path': "artifacts_mock",
            'location': "stockfish_mock",
            'threads': 4,
            'hash': 128,
            'minimum_thinking_time': 1,
            'raw_data_path': "raw_data_mock",
            'processed_path': "processed_data_mock",
            'engine_depth': 5,
            'mate_value': 10000
        }
        process_raw_data.init_worker(config)

        dfs_rows = []

        def mock_write_parquet_side_effect(df, file):
            nonlocal dfs_rows
            dfs_rows.append(df.rows())

        mock_write_parquet.side_effect = mock_write_parquet_side_effect

        process_raw_data.process_raw_data(config)

        mock_open.assert_called_with("artifacts_mock/svm.p", "rb")
        self.assertEqual(dfs_rows, [[('', '4', 'training', -10000, -10000, 1)], [('', '4', 'testing', -10000, -10000, 1)],
                                    [('', '4', 'validing', -10000, -10000, 1)]])