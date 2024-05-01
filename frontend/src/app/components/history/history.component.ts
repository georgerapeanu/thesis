import { AfterViewChecked, Component, ElementRef, OnDestroy, OnInit, ViewChild } from '@angular/core';
import { GameStateService } from '../../services/game-state.service';
import { CommonModule } from '@angular/common';
import { Subscription } from 'rxjs';

@Component({
  selector: 'app-history',
  standalone: true,
  imports: [CommonModule],
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

  constructor(
    private gameStateService: GameStateService
  ) {
    this.gameStateService = gameStateService;
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

    });
  }

  ngOnDestroy(): void {
    this.gameStateSubscription?.unsubscribe();
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
}
