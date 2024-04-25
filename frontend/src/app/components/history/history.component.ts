import { Component, OnInit } from '@angular/core';
import { GameStateService } from '../../services/game-state.service';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-history',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './history.component.html',
  styleUrl: './history.component.css'
})
export class HistoryComponent implements OnInit {
  moves_enumerated: Array<string> = [];
  moves_indexed: Array<number | null> = [];
  current_index: number = 0;

  constructor(
    private gameStateService: GameStateService
  ) {
    this.gameStateService = gameStateService;
  }

  ngOnInit(): void {
    this.gameStateService.get_observable_state().subscribe(game_index => {
      this.moves_enumerated = [];
      this.moves_indexed = [];
      this.current_index = 0;

      let [game, index] = game_index;
      this.current_index = index;
       game.history().forEach((move, i) => {
        if(this.moves_indexed.length % 3 == 0) {
          this.moves_enumerated.push(Math.round(this.moves_indexed.length / 3 + 1).toString() + ".");
          this.moves_indexed.push(null);
        }
        this.moves_enumerated.push(move);
        this.moves_indexed.push(i);
      });
    });
  }

  onClick(index: number | null) {
    if(index === null) {
      return;
    }
    console.log("clicked ", index);
    this.gameStateService.seek(index + 1);
  }
}
