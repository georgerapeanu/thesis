import chess
import flask
import hydra
import sentencepiece
import stockfish
from omegaconf import DictConfig, OmegaConf
from flask import Flask, request, Response
import json
from typing import *
import jsonschema

import torch

from data.ActualBoardCommentaryDataset import ActualBoardCommentaryDataset
from model.commentary_models import ActualBoardTransformerMultipleHeadsModel

engine = None

@hydra.main(config_path="./conf", config_name="serve_config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    app = Flask(__name__)
    model: ActualBoardTransformerMultipleHeadsModel = torch.jit.load(cfg["model_path"])
    sp = sentencepiece.SentencePieceProcessor(cfg["sentencepiece_path"])

    def evaluation_to_value(evaluation):
        if evaluation['type'] == 'cp':
            return evaluation['value']
        else:
            return cfg['engine_config']['mate_value'] if evaluation['value'] > 0 else -cfg['engine_config']['mate_value']

    def position_to_value(engine: stockfish.Stockfish, position: str):
        engine.set_fen_position(position)
        return evaluation_to_value(engine.get_evaluation())


    payload_schema = {
        "type": "object",
        "properties": {
            "past_boards": {
                "type": "array",
                "items": {"type": "string"}
            },
            "current_board": {
                "type": "string"
            },
            "temperature": {
                "type": "number"
            },
            "do_sample": {
                "type": "boolean"
            },
            "target_type": {
                "type": "string"
            },
            "max_new_tokens": {
                "type": "number"
            }
        },
        "required": ["past_boards", "current_board"]
    }

    TARGET_TYPES_TO_IDS = {
        'MoveDesc': 0,
        'MoveQuality': 1,
        'Comparative': 2,
        "Strategy": 3,
        "Context": 4
    }

    #TODO protocol for determining when sequence ends
    #TODO validation on payload
    @app.get('/annotate')
    def process():
        try:
            data = json.loads(request.data)
            jsonschema.validate(instance=data, schema=payload_schema)

            past_boards = data.get('past_boards')
            current_board = data.get('current_board')

            if len(past_boards) != cfg['count_past_boards']:
                return {"error": "Length of past boards array is not the same as the one configured in config"}

            for board in past_boards + [current_board]:
                try:
                    chess.Board(board)
                except ValueError:
                    raise ValueError("Board FEN is invalid")

            temperature = 1.0 if 'temperature' not in data else data.get('temperature')

            if temperature <= 0:
                raise ValueError("Temperature should be bigger than 0")

            do_sample = False if 'do_sample' not in data else data.get('do_sample')
            target_type = None if 'target_type' not in data else data.get('target_type')

            if target_type is not None and target_type not in TARGET_TYPES_TO_IDS:
                raise ValueError("Target type is not recognized")
            if target_type is not None:
                target_type = torch.tensor(TARGET_TYPES_TO_IDS[target_type])
            max_new_tokens = 1000 if 'max_new_tokens' not in data else data.get('max_new_tokens')

            if max_new_tokens <= 0:
                raise ValueError("max_new_tokens should be bigger than 0")
        except json.JSONDecodeError:
            return {"error": "JSON payload is malformed"}
        except jsonschema.ValidationError:
            return {"error": "JSON payload does not respect schema specification"}
        except ValueError as e:
            return {"error": str(e)}

        global engine
        if engine is None:
            engine = stockfish.Stockfish(cfg['engine_config']['location'], parameters={
                "Threads": cfg['engine_config']['threads'],
                "Hash": cfg['engine_config']['hash'],
                "Minimum Thinking Time": cfg['engine_config']['minimum_thinking_time']
            })
            engine.set_depth(cfg['engine_config']['engine_depth'])

        (X_board, X_strength, X_reps, X_state,  _, _) = ActualBoardCommentaryDataset.raw_data_to_data(
            (
                list(zip(past_boards, list(map(lambda x: position_to_value(engine, x), past_boards)))),
                (current_board, position_to_value(engine, current_board)),
                torch.zeros(0),
                torch.zeros(0)
            ),
            cfg['count_past_boards'],
            cfg['engine_config']['mate_value']
        )
        X_text = torch.tensor([sp.bos_id()])
        X_text = X_text.unsqueeze(0)
        X_board = X_board.unsqueeze(0)
        X_strength = X_strength.unsqueeze(0)
        X_reps = X_reps.unsqueeze(0)
        X_state = X_state.unsqueeze(0)

        def generate():
            nonlocal X_text
            nonlocal X_text
            nonlocal X_board
            nonlocal X_strength
            nonlocal X_reps
            nonlocal X_state
            with torch.no_grad():
                for i in range(max_new_tokens):
                    X_text = X_text if X_text.size(1) < cfg.context_length else X_text[:, -cfg.context_length:]
                    logits, _ = model(X_board, X_strength, X_reps, X_state, X_text,
                                      (torch.zeros(1, X_text.size(1)) == 1).to(X_board.device), target_type=target_type)
                    logits = logits[:, -1, :] / temperature
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    if do_sample is not False:
                        text_next = torch.multinomial(probs, num_samples=1)
                    else:
                        _, text_next = torch.topk(probs, k=1, dim=-1)
                    X_text = torch.cat([X_text, text_next], dim=1)
                    if text_next == model.eos_id:
                        break
                    skip_length = 0
                    if X_text.size(1) > 1:
                        skip_length = len(sp.decode(X_text[0, -2].view(-1).tolist()).replace("<n>", "\n"))
                    yield sp.decode(X_text[0, -2:].view(-1).tolist()).replace("<n>", "\n")[skip_length:]
                    # yield sp.decode(text_next.view(-1).tolist()).replace("<n>", "\n") + "\n"
                    # yield sp.decode(X_text[0, :-2].tolist()).replace("<n>", "\n")[skip_length:]
        return app.response_class(generate(), mimetype='text')
    app.run(host="0.0.0.0", port=8080)


if __name__ == '__main__':
    main()