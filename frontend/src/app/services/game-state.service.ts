import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';
import { Chess, Move, validateFen } from 'chess.js';

@Injectable({
  providedIn: 'root'
})
export class GameStateService {
  private current_game = new BehaviorSubject<Chess>(new Chess());

  constructor() { }

  set_current_fen(new_fen: string): void | Error {
    var validationResult = validateFen(new_fen);
    if(!validationResult['ok']) {
      return new Error(validationResult["error"]);
    }

    this.current_game.next(new Chess(new_fen));
  }

  get_observable_game(): Observable<Chess> {
    return this.current_game.asObservable();
  }

  get_current_state(): Chess {
    return this.current_game.value;
  }

  move(board_move: Move) {
    this.current_game.value.move(board_move);
    this.current_game.next(this.current_game.value);
  }
}
