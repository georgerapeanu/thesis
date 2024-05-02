import { AfterViewChecked, Component, ElementRef, EventEmitter, OnDestroy, OnInit, Output, ViewChild } from '@angular/core';
import { GameStateService } from '../../services/game-state.service';
import { CommonModule } from '@angular/common';
import { Subscription } from 'rxjs';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import { ChessEngineService } from '../../services/chess-engine.service';
import { EvaluationDTO } from '../../dto/evaluationDTO';

@Component({
  selector: 'app-history',
  standalone: true,
  imports: [CommonModule, MatButtonModule, MatIconModule],
  templateUrl: './history.component.html',
  styleUrl: './history.component.css'
})
export class HistoryComponent implements OnInit, AfterViewChecked, OnDestroy {

  moves_enumerated: Array<string> = [];
  moves_indexed: Array<number | null> = [];
  current_index: number = 0;
  @ViewChild("toFocus")
  toFocus: ElementRef | undefined = undefined;
  gameStateSubscription: Subscription | null = null;
  evaluationSubscription: Subscription | null = null;
  lastEvaluation: EvaluationDTO | null = null;
  evaluationPending = false;

  @Output() requestFlip = new EventEmitter<null>();

  constructor(
    private gameStateService: GameStateService,
    private chessEngineService: ChessEngineService
  ) {
    this.gameStateService = gameStateService;
    this.chessEngineService = chessEngineService;
  }

  ngOnInit(): void {
    this.gameStateSubscription = this.gameStateService.get_observable_state().subscribe(game_index => {
      this.moves_enumerated = [];
      this.moves_indexed = [];
      this.current_index = 0;

      let [game, index] = game_index;
      this.current_index = index;
       game.history().forEach((move, i) => {
        if(this.moves_indexed.length % 3 == 0) {
          this.moves_enumerated.push(Math.round(this.moves_indexed.length / 3 + 1).toString());
          this.moves_indexed.push(null);
        }
        this.moves_enumerated.push(move);
        this.moves_indexed.push(i);
      });

      this.evaluationSubscription?.unsubscribe();
      this.evaluationPending = true;
      this.evaluationSubscription = this.chessEngineService.requestEvaluation(this.gameStateService.get_chess_game_at_current_index(0))
      .subscribe((evaluation) => {
        this.evaluationPending = false;
        this.lastEvaluation = evaluation;
      });

    });
  }

  ngOnDestroy(): void {
    this.gameStateSubscription?.unsubscribe();
    this.evaluationSubscription?.unsubscribe();
  }

  ngAfterViewChecked(): void {
    if(this.toFocus) {
      this.toFocus.nativeElement.parentElement.scrollIntoView({block: 'center', behavior: 'smooth'});
    }
  }

  onClick(index: number | null) {
    if(index === null) {
      return;
    }
    this.gameStateService.seek(index + 1);
  }

  onRequestFlip() {
    this.requestFlip.emit(null);
  }
}
