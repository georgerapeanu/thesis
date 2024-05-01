import { Component, OnDestroy, OnInit } from '@angular/core';
import { GameStateService } from '../../services/game-state.service';
import { Chess, validateFen } from 'chess.js';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { Subscription } from 'rxjs';

@Component({
  selector: 'app-game-state',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './game-state.component.html',
  styleUrl: './game-state.component.css'
})
export class GameStateComponent implements OnInit, OnDestroy {
  pgn: string = "";
  fen: string = "";
  fen_invalid: boolean = false;
  pgn_invalid: boolean = false;
  gameStateSubscription: Subscription | null = null

  constructor(
    private gameStateService: GameStateService
  ) {
    this.gameStateService = gameStateService;
  }

  ngOnDestroy(): void {
    this.gameStateSubscription?.unsubscribe();
  }

  ngOnInit(): void {
    this.gameStateSubscription = this.gameStateService.get_observable_state().subscribe((game_index: [Chess, number]): void => {
      this.pgn = game_index[0].pgn();
      this.fen = this.gameStateService.get_chess_game_at_index(0).fen();
      this.fen_invalid = false;
      this.pgn_invalid = false;
    });
  }

  onEnterFEN(): void {
    if(validateFen(this.fen).ok === true) {
      this.gameStateService.set_current_fen(this.fen);
      return ;
    }
    this.fen_invalid = true;
  }

  onEnterPGN(): void {
    let maybe_error = this.gameStateService.set_pgn(this.pgn);
    if(maybe_error instanceof Error) {
      this.pgn_invalid = true;
    }
  }

  onFocusOut(): void {
    this.pgn = this.gameStateService.get_current_state()[0].pgn();
    this.fen = this.gameStateService.get_chess_game_at_index(0).fen();
    this.fen_invalid = false;
    this.pgn_invalid = false;
  }

}
