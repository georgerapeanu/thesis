import { Component, OnInit } from '@angular/core';
import { MatGridListModule } from '@angular/material/grid-list';
import {MatButtonModule} from '@angular/material/button';
import { ModelBackendService } from '../../services/model-backend.service';
import { GameStateService } from '../../services/game-state.service';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-commentary',
  standalone: true,
  imports: [CommonModule, MatGridListModule, MatButtonModule],
  templateUrl: './commentary.component.html',
  styleUrl: './commentary.component.css'
})
export class CommentaryComponent implements OnInit {

  public raw_commentary = "";
  public prefix = "";

  constructor(private modelBackendService: ModelBackendService, private gameStateService: GameStateService) {
    this.modelBackendService = modelBackendService;
    this.gameStateService = gameStateService;
  }

  ngOnInit(): void {
    this.modelBackendService.getPrefixObservable().subscribe({
      next: (prefix) => {
        this.prefix = prefix;
        this.raw_commentary = "";
      }
    });
  }

  request_commentary(): void {
    this.raw_commentary = "";
    this.modelBackendService.getAnnotation(this.gameStateService.get_chess_game_at_index(2)).subscribe((value) => {
      this.raw_commentary += value;
    });
  }

  get commentary(): string {
    return this.raw_commentary.replace("<n>", "\n");
  }
}
