import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { BoardComponent } from './components/board/board.component';
import { ModelBackendService } from './services/model-backend.service';
import { GameStateService } from './services/game-state.service';
import { HttpClientModule } from '@angular/common/http';
import { GameStateComponent } from './components/game-state/game-state.component';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, BoardComponent, FormsModule, HttpClientModule, GameStateComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'frontend';

  modelBackendService: ModelBackendService;
  gameStateService: GameStateService;
  commentary: string = "";

  constructor(
    modelBackendService: ModelBackendService,
    gameStateService: GameStateService
  ) {
    this.modelBackendService = modelBackendService;
    this.gameStateService = gameStateService;
  }

  public requestCommentary() {
    this.commentary = "";
    this.modelBackendService.getAnnotation(this.gameStateService.get_chess_game_at_index()).subscribe({
      next: (value => this.commentary += value),
      error: (value => alert(value))
    });
  }
}
