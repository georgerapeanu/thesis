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

  private _temperature = 1;
  public get temperature() {
    return this._temperature;
  }
  public set temperature(value) {
    this._temperature = value;
  }

  private _doSample = false;
  public get doSample() {
    return this._doSample;
  }
  public set doSample(value) {
    this._doSample = value;
  }

  private _commentary_type = "";
  public get commentary_type() {
    return this._commentary_type;
  }
  public set commentary_type(value) {
    this._commentary_type = value;
  }

  private _max_new_tokens = 1000;
  public get max_new_tokens() {
    return this._max_new_tokens;
  }
  public set max_new_tokens(value) {
    this._max_new_tokens = value;
  }

  private _prefix = "";
  public get prefix() {
    return this._prefix;
  }
  public set prefix(value) {
    this._prefix = value;
  }

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
        "do_sample": true,
        "temperature": 0.3
      })
    })
    .pipe(switchMap((result) => {
      return from(result.text());
    }));
  }
}
