import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { BoardComponent } from './components/board/board.component';
import { ModelBackendService } from './services/model-backend.service';
import { GameStateService } from './services/game-state.service';
import { HttpClientModule } from '@angular/common/http';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, BoardComponent, HttpClientModule],
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
    console.log("called request");
    this.commentary = "";
    this.modelBackendService.getAnnotation(this.gameStateService.get_current_state()).subscribe({
      next: (value => this.commentary += value),
      error: (value => alert(value))
    });
  }
}
