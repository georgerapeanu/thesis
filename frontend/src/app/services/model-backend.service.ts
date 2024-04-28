import { HttpClient, HttpDownloadProgressEvent, HttpEvent, HttpEventType } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { environment } from '../..//environments/environment';
import { GameStateService } from './game-state.service';
import { Chess } from 'chess.js';
import { Observable, map, switchMap, from, BehaviorSubject, ReadableStreamLike} from 'rxjs';
import { fromFetch } from "rxjs/fetch";

interface AnnotateRequestDict {
  past_boards: any;
  current_board: any;
  temperature?: any;
  do_sample: boolean;
  target_type?: string;
  max_new_tokens: number;
  prefix?: string;
}


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

  private _doSample = true;
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
    return this._prefix.replace("<n>", "\n");
  }
  public set prefix(value) {
    this._prefix = value.replace("\n", "<n>");
    this.prefix_behvaior_subject.next(this.prefix);
  }

  private prefix_behvaior_subject = new BehaviorSubject(this.prefix);
  private decoder = new TextDecoder("utf-8");

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
    var request_dict: AnnotateRequestDict  = {
      "past_boards": past_boards,
      "current_board": current_board,
      "do_sample": this._doSample,
      "max_new_tokens": this._max_new_tokens,
    };
    if(this.doSample) {
      request_dict["temperature"] = this._temperature;
    }
    if(this.commentary_type.length > 0) {
      request_dict["target_type"] = this._commentary_type;
    }
    if(this.prefix.length > 0) {
      request_dict["prefix"] = this._prefix;
    }
    return fromFetch(environment.modelURL + "/get_commentary", {
      method: "POST",
      body: JSON.stringify(request_dict)
    })
    .pipe(switchMap((result) => {
      if(!result.body) {
        throw "Error reading response";
      }
      return from(result.body! as ReadableStreamLike<Uint8Array>);
    }))
    .pipe(map(bytes => {
      return this.decoder.decode(bytes);
    }));
  }

  getPrefixObservable(): Observable<string> {
    return this.prefix_behvaior_subject.asObservable();
  }
}
