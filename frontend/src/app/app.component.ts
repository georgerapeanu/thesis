import { Component, OnDestroy, OnInit } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { BoardComponent } from './components/board/board.component';
import { ModelBackendService } from './services/model-backend.service';
import { GameStateService } from './services/game-state.service';
import { HttpClientModule } from '@angular/common/http';
import { GameStateComponent } from './components/game-state/game-state.component';
import { HistoryComponent } from './components/history/history.component';
import { ModelSettingsComponent } from './components/model-settings/model-settings.component';
import { MatGridListModule } from '@angular/material/grid-list';
import { CommentaryComponent } from './components/commentary/commentary.component';
import { SeeTopkComponent } from './components/see-topk/see-topk.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, SeeTopkComponent, BoardComponent, HistoryComponent, ModelSettingsComponent, CommentaryComponent, HttpClientModule, GameStateComponent, MatGridListModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent implements OnInit, OnDestroy {
  title = 'frontend';

  modelBackendService: ModelBackendService;
  gameStateService: GameStateService;
  commentary: string = "";
  flipped: boolean = false;

  constructor(
    modelBackendService: ModelBackendService,
    gameStateService: GameStateService
  ) {
    this.modelBackendService = modelBackendService;
    this.gameStateService = gameStateService;
  }

  ngOnDestroy(): void {
  }

  ngOnInit(): void {
  }


  public onRequestFlip() {
    this.flipped = !this.flipped;
  }
}
