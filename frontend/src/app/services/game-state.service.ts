import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';
import { Chess, Move, validateFen } from 'chess.js';

@Injectable({
  providedIn: 'root'
})
export class GameStateService {
  private current_game = (new Chess());
  private move_index= 0;
  private subject = new BehaviorSubject<[Chess, number]>([this.current_game, this.move_index]);
  constructor() { }

  set_current_fen(new_fen: string): Error | null {
    var validationResult = validateFen(new_fen);
    if(!validationResult['ok']) {
      return new Error(validationResult["error"]);
    }
    this.current_game = new Chess(new_fen);
    this.move_index = 0;
    this.subject.next([this.current_game, this.move_index]);
    return null;
  }

  set_pgn(pgn: string): Error | null {
    try {
      let new_game = new Chess();
      new_game.loadPgn(pgn);
      this.current_game = new_game;
      this.move_index = new_game.history().length;
      this.subject.next([this.current_game, this.move_index]);
      return null;
    } catch (exception) {
      return new Error("Invalid pgn");
    }
  }

  get_observable_state(): Observable<[Chess, number]> {
    return this.subject.asObservable();
  }

  get_current_state(): [Chess, number] {
    return this.subject.value;
  }

  move(board_move: Move) {
    while(this.current_game.history().length > this.move_index) {
      this.current_game.undo();
    }
    this.current_game.move(board_move);
    this.move_index += 1;
    this.subject.next([this.current_game, this.move_index]);
  }

  seek(move_index: number): void | Error {
    if(move_index < 0 || move_index > this.current_game.history().length + 1) {
      return new Error("move index is invalid");
    }
    this.move_index = move_index;
    this.subject.next([this.current_game, this.move_index]);
  }

  undo(): void | Error {
    return this.seek(this.move_index - 1);
  }

  get_chess_game_at_index(): Chess {
    let chess_game_at_index: Chess = new Chess();
    chess_game_at_index.loadPgn(this.current_game.pgn());
    while(chess_game_at_index.history().length > this.move_index) {
      chess_game_at_index.undo();
    }
    return chess_game_at_index;
  }
}
