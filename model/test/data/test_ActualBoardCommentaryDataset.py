import unittest

import chess
import torch

from data.ActualBoardCommentaryDataset import ActualBoardCommentaryDataset


class TestActualBoardCommentaryDataset(unittest.TestCase):
    def test_get_board_token_size(self):
        self.assertEqual(ActualBoardCommentaryDataset.get_board_token_size(), 13)

    def test_get_state_features(self):
        self.assertTrue(torch.equal(
            torch.tensor([ 0., 35.,  0.,  0.,  0.,  0.,  1.]),
            ActualBoardCommentaryDataset.get_state_features(chess.Board("8/R5pk/1p6/2p4p/4b1qP/P1Q3B1/1P3PP1/3r2K1 w - - 1 35"))
        ))

    def test_get_positional_features(self):
        board_stuff, repetitions = (ActualBoardCommentaryDataset.get_positional_features(chess.Board("8/R5pk/1p6/2p4p/4b1qP/P1Q3B1/1P3PP1/3r2K1 w - - 1 35")))
        self.assertTrue(torch.equal(
            torch.tensor([ 0,  0,  0,  8,  0,  0,  3,  0,  0,  1,  0,  0,  0,  1,  1,  0,  1,  0,
             5,  0,  0,  0,  4,  0,  0,  0,  0,  0, 10,  0, 11,  1,  0,  0,  7,  0,
             0,  0,  0,  7,  0,  7,  0,  0,  0,  0,  0,  0,  2,  0,  0,  0,  0,  0,
             7,  9,  0,  0,  0,  0,  0,  0,  0,  0], dtype=torch.int32),
            board_stuff
        ))
        self.assertEqual(repetitions, 1)

    def test_raw_data_to_data(self):
        (boards_tensor,
        strengths_tensor,
        reps_tensor,
        state_tensor,
        tokens,
        types) = ActualBoardCommentaryDataset.raw_data_to_data(
            (
                [('8/R5pk/1p6/2pr3p/4b1qP/P1Q3B1/1P3PP1/6K1 b - - 0 34', 500)],
                ('8/R5pk/1p6/2p4p/4b1qP/P1Q3B1/1P3PP1/3r2K1 w - - 1 35', -40),
                [2, 4, 3],
                torch.tensor([0, 0, 1, 1, 0, 1])
            ),
            2,
            10000
        )
        self.assertTrue(torch.equal(
            boards_tensor,
            torch.tensor([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  3,  0,  0,  1,  0,  0,  0,  1,  1,  0,  1,  0,
              5,  0,  0,  0,  4,  0,  0,  0,  0,  0, 10,  0, 11,  1,  0,  0,  7,  8,
              0,  0,  0,  7,  0,  7,  0,  0,  0,  0,  0,  0,  2,  0,  0,  0,  0,  0,
              7,  9,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  8,  0,  0,  3,  0,  0,  1,  0,  0,  0,  1,  1,  0,  1,  0,
              5,  0,  0,  0,  4,  0,  0,  0,  0,  0, 10,  0, 11,  1,  0,  0,  7,  0,
              0,  0,  0,  7,  0,  7,  0,  0,  0,  0,  0,  0,  2,  0,  0,  0,  0,  0,
              7,  9,  0,  0,  0,  0,  0,  0,  0,  0]], dtype=torch.int32)
        ))
        self.assertTrue(torch.equal(
            strengths_tensor,
            torch.tensor([[0.0000],
                    [0.0500],
                    [-0.0040]])
        ))
        self.assertTrue(torch.allclose(
            reps_tensor,
            torch.tensor([[0.0000],
                    [0.3333],
                    [0.3333]]),
            atol=1e-4,
            rtol=1e-4
        ))
        self.assertTrue(torch.equal(
            state_tensor,
            torch.tensor([0., 35., 0., 0., 0., 0., 1.])
        ))
        self.assertTrue(torch.equal(
            tokens,
            torch.tensor([2, 4, 3])
        ))
        self.assertTrue(torch.equal(
            types,
            torch.tensor([0, 0, 1, 1, 0, 1])
        ))



if __name__ == '__main__':
    unittest.main()
