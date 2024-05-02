import { Component, OnDestroy, OnInit } from '@angular/core';
import { MatGridListModule } from '@angular/material/grid-list';
import {MatButtonModule} from '@angular/material/button';
import { ModelBackendService } from '../../services/model-backend.service';
import { GameStateService } from '../../services/game-state.service';
import { CommonModule } from '@angular/common';
import { Subscription } from 'rxjs';

@Component({
  selector: 'app-commentary',
  standalone: true,
  imports: [CommonModule, MatGridListModule, MatButtonModule],
  templateUrl: './commentary.component.html',
  styleUrl: './commentary.component.css'
})
export class CommentaryComponent implements OnInit, OnDestroy {

  public raw_commentary = "";
  public prefix = "";
  public is_placeholder = true;
  prefixSubscription: Subscription | null = null;
  annotateSubscription: Subscription | null = null;

  constructor(private modelBackendService: ModelBackendService, private gameStateService: GameStateService) {
    this.modelBackendService = modelBackendService;
    this.gameStateService = gameStateService;
  }

  ngOnInit(): void {
    this.prefixSubscription = this.modelBackendService.getPrefixObservable().subscribe({
      next: (prefix) => {
        this.prefix = prefix;
        this.raw_commentary = "";
        this.is_placeholder = true;
      }
    });
  }

  ngOnDestroy(): void {
    this.prefixSubscription?.unsubscribe();
    this.annotateSubscription?.unsubscribe();
  }

  request_commentary(): void {
    this.raw_commentary = "";
    this.is_placeholder = true;
    if(this.annotateSubscription) {
      this.annotateSubscription.unsubscribe();
    }
    this.annotateSubscription = this.modelBackendService.getAnnotation(this.gameStateService.get_chess_game_at_current_index(2)).subscribe({
      next: (value) => {
        this.raw_commentary += value;
        this.is_placeholder = false;
      },
      complete: () => {
        this.is_placeholder = false;
      }
    });
  }

  get commentary(): string {
    return this.raw_commentary.replace("<n>", "\n");
  }
}
