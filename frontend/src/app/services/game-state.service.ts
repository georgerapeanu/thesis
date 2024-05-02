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
    for(let i = this.current_game.history().length; i > this.move_index; i--) {
      this.current_game.undo();
    }
    this.current_game.move(board_move);
    this.move_index += 1;
    this.subject.next([this.current_game, this.move_index]);
  }

  seek(move_index: number): null | Error {
    if(move_index < 0 || move_index > this.current_game.history().length) {
      return new Error("move index is invalid");
    }
    this.move_index = move_index;
    this.subject.next([this.current_game, this.move_index]);
    return null;
  }

  undo(): null | Error {
    return this.seek(this.move_index - 1);
  }

  redo(): null | Error {
    return this.seek(this.move_index + 1);
  }

  get_chess_game_at_current_index(min_history: number): Chess {
    if(this.move_index > 0) {
      if(this.move_index - 1 - min_history >= 0) {
        var chess_game_at_index= new Chess(this.current_game.history({verbose: true})[this.move_index - 1 - min_history].after);
      } else {
        var chess_game_at_index = new Chess();
      }
      let moves = this.current_game.history().slice(Math.max(0, this.move_index - min_history), this.move_index);
      for(const move of moves) {
        chess_game_at_index.move(move);
      }
    } else if(this.current_game.history().length > 0) {
      var chess_game_at_index = new Chess(this.current_game.history({verbose: true})[0].before);
    } else {
      var chess_game_at_index = this.current_game;
    }
    return chess_game_at_index;
  }
}
