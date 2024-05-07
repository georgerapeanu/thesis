import { TestBed } from '@angular/core/testing';

import { GameStateService } from './game-state.service';
import { first, skip } from 'rxjs';
import { Chess } from 'chess.js';

describe('GameStateService', () => {
  let service: GameStateService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(GameStateService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should set fen and notify observers', () => {
    const fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    service.get_observable_state().pipe(skip(1), first()).subscribe((game_index) => {
      expect(game_index[0].fen()).toEqual(fen);
      expect(service.get_chess_game_at_current_index(0).fen()).toEqual(fen);
    });
    expect(service.set_current_fen(fen)).toBeNull();
  });

  it('should ignore invalid fen', () => {
    const fen = "invalid";
    let test_function = (_value: [Chess, number]) => {};
    let test = jasmine.createSpy().and.callFake(test_function);
    service.get_observable_state().pipe(skip(1), first()).subscribe(test);
    expect(service.set_current_fen(fen)).toBeInstanceOf(Error);
    expect(test).not.toHaveBeenCalled();
  });

  it('should set pgn and notify observers', () => {
    const pgn = "1. e4 Nf6 2. e5";
    const fen = "rnbqkb1r/pppppppp/5n2/4P3/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2";
    service.get_observable_state().pipe(skip(1), first()).subscribe((game_index) => {
      expect(game_index[0].fen()).toEqual(fen);
      expect(service.get_chess_game_at_current_index(0).fen()).toEqual(fen);
      expect(game_index[0].history()).toEqual(['e4', 'Nf6', 'e5']);
    });
    expect(service.set_pgn(pgn)).toBeNull();
  });

  it('should ignore invalid pgn', () => {
    const pgn = "invalid";
    let test_function = (_value: [Chess, number]) => {};
    let test = jasmine.createSpy().and.callFake(test_function);
    service.get_observable_state().pipe(skip(1), first()).subscribe(test);
    expect(service.set_pgn(pgn)).toBeInstanceOf(Error);
    expect(test).not.toHaveBeenCalled();
  });

  it('should move and notify observers', () => {
    const fen = "rnbqkb1r/pppppppp/5n2/4P3/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2";
    service.set_current_fen(fen);

    service.get_observable_state().pipe(skip(1), first()).subscribe((game_index) => {
      expect(game_index[0].fen()).toEqual("rnbqkb1r/pppppppp/8/3nP3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 1 3");
      expect(service.get_chess_game_at_current_index(0).fen()).toEqual("rnbqkb1r/pppppppp/8/3nP3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 1 3");
    });
    let chess = new Chess(fen);
    chess.move('Nd5');
    let move = chess.history({verbose: true})[0];
    service.move(move);

  });

  it('should undo and notify observers', () => {
    const fen = "rnbqkb1r/pppppppp/5n2/4P3/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2";
    service.set_current_fen(fen);

    let chess = new Chess(fen);
    chess.move('Nd5');
    let move = chess.history({verbose: true})[0];
    service.move(move);

    service.get_observable_state().pipe(skip(1), first()).subscribe((_game_index) => {
      expect(service.get_chess_game_at_current_index(0).fen()).toEqual(fen);
    });
    service.undo();
  });

  it('should redo and notify observers', () => {
    const fen = "rnbqkb1r/pppppppp/5n2/4P3/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2";
    service.set_current_fen(fen);

    let chess = new Chess(fen);
    chess.move('Nd5');
    let move = chess.history({verbose: true})[0];
    service.move(move);
    service.undo();

    service.get_observable_state().pipe(skip(1), first()).subscribe((_game_index) => {
      expect(service.get_chess_game_at_current_index(0).fen()).toEqual("rnbqkb1r/pppppppp/8/3nP3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 1 3");
    });
    service.redo();
  });

  it('should return an error on invalid seek', () => {
    expect(service.seek(10)).toBeInstanceOf(Error);
  });

  it('should return null on valid seek', () => {
    expect(service.seek(0)).toBeNull();
  });

  it('should return copy with desired history', () => {
    const pgn = "1. e4 Nf6 2. e5";
    service.set_pgn(pgn);
    service.undo();
    let game = service.get_chess_game_at_current_index(0);
    expect(game.history()).toEqual([]);
    expect(game.fen()).toEqual('rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2');

    game = service.get_chess_game_at_current_index(3);
    expect(game.history()).toEqual(["e4", "Nf6"]);
    expect(game.fen()).toEqual('rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2');
  });

  it('should return current state', () => {
    const pgn = "1. e4 Nf6 2. e5";
    service.set_pgn(pgn);
    let game_index = service.get_current_state();
    let game = game_index[0];
    expect(game_index[1]).toEqual(3);
    expect(game.history()).toEqual(["e4", "Nf6", "e5"]);
    expect(game.fen()).toEqual('rnbqkb1r/pppppppp/5n2/4P3/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2');
  });

  it('should override moves if moved after seek/undo', () => {
    const pgn = "1. e4 Nf6 2. e5";
    service.set_pgn(pgn);
    service.undo();
    let chess = new Chess();
    chess.move('e4');
    chess.move('Nf6');
    chess.move('d4');
    let move = chess.history({verbose: true})[2];
    service.move(move);
    let game_index = service.get_current_state();
    let game = game_index[0];
    expect(game_index[1]).toEqual(3);
    expect(game.history()).toEqual(["e4", "Nf6", "d4"]);
    expect(game.fen()).toEqual('rnbqkb1r/pppppppp/5n2/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2');
  });
});
