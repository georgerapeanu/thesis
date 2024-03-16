## idea

use stockfish to get scores
use some sort of nn to translate those to winning probabilities
rollout a few games(best moves, moves that look good up to a depth but fall off)
take games as input, transformer them


input format

for a position:
input mask : 8 x 8 x 109 -> conv layer for features 8 x 8 x 256
8 x 8 x  71(types of moves) x (D + 1 + 3) (the strengh at D depths, the true strengh, is capture, is attack, is check) -> linear layer 8 x 8 x 71 x F 


add attacks for both sides plane(maybe)

previous board state is always included
some playouts need to be put in 
lets say we have an array for the number of moves explored at each step
these are in predfs order

add positional encodings to these 
and attention to these 

add the text and attention to it too 

preprocessing: transform all chess moves to algebraic notation

possible moves like in alpha zero output format + threats + attacks(maybe each move would have its score for each depth, 1, 2, ...)
maybe have lines containing initial state, moves that would lead to it(based on rollouts), and the state descirbed above
all those fed through a transformer
output:
probability distribution for next token


