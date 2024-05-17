import unittest
from unittest import mock
import polars as pl

from data import extract_commentaries_for_spm


class TestExtractCommentariesForSVM(unittest.TestCase):

    @mock.patch('data.extract_commentaries_for_spm.pickle.load')
    @mock.patch('data.extract_commentaries_for_spm.open')
    def test_extract(self,
                    mock_open: mock.Mock,
                    mock_pickle_load: mock.Mock

    ) -> None:
        mock_f = mock.MagicMock()
        mock_g = mock.MagicMock()

        path, mode = None, None
        def enter_side_effect():
            nonlocal path
            nonlocal mode
            nonlocal mock_f
            nonlocal mock_g
            if mode is None:
                self.assertEqual(path, "artifacts_path_mock/commentaries_raw.txt")
                return mock_g
            elif mode == 'w':
                self.assertEqual(path, "artifacts_path_mock/commentaries.txt")
                return mock_f
            else:
                self.assertEqual(True, False, "Unexpected mode {}".format(mode))

        def mock_open_side_effect(_path, _mode = None):
            nonlocal  path, mode
            path = _path
            mode = _mode
            return mock_open
        mock_open.side_effect = mock_open_side_effect
        mock_open.__enter__.side_effect = enter_side_effect

        (vectorizer_mock, classifiers_mock) = (mock.Mock(), [mock.MagicMock(), mock.MagicMock()])
        classifiers_mock[0].predict.return_value = [0, 1, 0, 0]
        classifiers_mock[1].predict.return_value = [1, 0, 0, 1]

        mock_pickle_load.return_value = (vectorizer_mock, classifiers_mock)

        mock_g.__iter__.return_value = iter(['a\n', 'b\n', 'c\n', 'd\n'])
        extract_commentaries_for_spm.extract_spm("artifacts_path_mock")

        mock_f.write.assert_has_calls([
            mock.call('b\n')
        ])




