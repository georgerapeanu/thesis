import { Component, HostListener, OnInit } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { BoardComponent } from './components/board/board.component';
import { ModelBackendService } from './services/model-backend.service';
import { GameStateService } from './services/game-state.service';
import { HttpClientModule } from '@angular/common/http';
import { GameStateComponent } from './components/game-state/game-state.component';
import { HistoryComponent } from './components/history/history.component';
import { Subject, debounceTime } from 'rxjs';
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
export class AppComponent implements OnInit {
  title = 'frontend';

  modelBackendService: ModelBackendService;
  gameStateService: GameStateService;
  commentary: string = "";
  keyCommandObservable = new Subject<string>;

  constructor(
    modelBackendService: ModelBackendService,
    gameStateService: GameStateService
  ) {
    this.modelBackendService = modelBackendService;
    this.gameStateService = gameStateService;
  }

  ngOnInit(): void {
    this.keyCommandObservable
      .pipe(debounceTime(50))
      .subscribe((key) => {
        switch(key) {
          case 'ArrowLeft': this.undo(); break;
          case 'ArrowRight': this.redo(); break;
          case 'ArrowUp': this.top(); break;
          case 'ArrowDown': this.bottom(); break;
        }
      });
  }

  public undo() {
    this.gameStateService.undo();
  }

  public redo() {
    this.gameStateService.redo();
  }

  public top() {
    this.gameStateService.seek(0);
  }

  public bottom() {
    this.gameStateService.seek(this.gameStateService.get_current_state()[0].history().length);
  }

  @HostListener('document:keydown', ['$event'])
  handleKeyboardEvent(event: KeyboardEvent): void {
    this.keyCommandObservable.next(event.key);
  }
}
