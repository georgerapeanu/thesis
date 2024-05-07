import { Component, OnDestroy, OnInit } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { ModelBackendService } from '../../services/model-backend.service';
import { GameStateService } from '../../services/game-state.service';
import { CommonModule } from '@angular/common';
import { Subscription, map } from 'rxjs';
import { ModelSettingsDTO } from '../../dto/modelSettingsDTO';

@Component({
  selector: 'app-commentary',
  standalone: true,
  imports: [CommonModule, MatButtonModule],
  templateUrl: './commentary.component.html',
  styleUrl: './commentary.component.css'
})
export class CommentaryComponent implements OnInit, OnDestroy {

  public raw_commentary = "";
  public prefix = "";
  public error: String | null = null;
  public is_placeholder = true;
  prefixSubscription: Subscription | null = null;
  annotateSubscription: Subscription | null = null;

  constructor(private modelBackendService: ModelBackendService, private gameStateService: GameStateService) {
    this.modelBackendService = modelBackendService;
    this.gameStateService = gameStateService;
  }

  ngOnInit(): void {
    this.prefixSubscription = this.modelBackendService.getModelSettingsDistinctUntilChangedObservable()
    .pipe(map((settings: ModelSettingsDTO) => settings.prefix))
    .subscribe({
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
        this.error = null;
        this.raw_commentary += value;
        this.is_placeholder = false;
      },
      error: (_) => {
        this.error = "The request has failed.";
        this.is_placeholder = false;
      },
      complete: () => {
        this.error = null;
        this.is_placeholder = false;
      }
    });
  }

  get commentary(): string {
    return this.raw_commentary.replaceAll('<n>', '\n');
  }
}
