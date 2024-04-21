import { HttpClient, HttpDownloadProgressEvent, HttpEvent, HttpEventType } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { environment } from '../..//environments/environment';
import { GameStateService } from './game-state.service';
import { Chess } from 'chess.js';
import { Observable, map, switchMap, from} from 'rxjs';
import { fromFetch } from "rxjs/fetch";
@Injectable({
  providedIn: 'root'
})
export class ModelBackendService {

  constructor(
    private httpClient: HttpClient,
    private gameStateService: GameStateService
  ) {

    this.httpClient = httpClient;
    this.gameStateService = gameStateService;
  }

  getAnnotation(chess: Chess): Observable<string> {
    var current_board = chess.fen();
    var past_boards = chess.history({verbose: true}).slice(-2).map((move) => move.before);
    return fromFetch(environment.modelURL + "/annotate", {
      method: "POST",
      body: JSON.stringify({
        "past_boards": past_boards,
        "current_board": current_board,
      })
    })
    .pipe(switchMap((result) => {
      return from(result.text());
    }));
  }
}
