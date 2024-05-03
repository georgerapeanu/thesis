import { Injectable } from '@angular/core';
import { environment } from '../../environments/environment';
import { GameStateService } from './game-state.service';
import { Chess } from 'chess.js';
import { Observable, map, switchMap, from, BehaviorSubject, ReadableStreamLike, Subject, merge, debounceTime, retry, Subscription, distinctUntilChanged } from 'rxjs';
import { fromFetch } from "rxjs/fetch";
import { TopKDTO } from '../dto/topkDTO';
import { ModelSettingsDTO } from '../dto/modelSettingsDTO';

interface AnnotateRequestDict {
  past_boards: Array<string>;
  current_board: string;
  temperature?: number;
  do_sample: boolean;
  target_type?: string;
  max_new_tokens: number;
  prefix?: string;
}

interface TopKRequestDict {
  past_boards: Array<string>;
  current_board: string;
  target_type?: string;
  prefix?: string;
  temperature?: number;
}



@Injectable({
  providedIn: 'root'
})
export class ModelBackendService {


  private model_settings: ModelSettingsDTO = new ModelSettingsDTO({temperature: 1, do_sample: true, target_type: "", max_new_tokens: 1000, prefix: ""});
  private decoder = new TextDecoder("utf-8");
  private topk_behavior_subject = new BehaviorSubject<TopKDTO>(new TopKDTO([], TopKDTO.State.LOADING));
  private last_topk_subscription: Subscription | null = null;

  private model_settings_subject = new Subject<ModelSettingsDTO>();
  private distinct_until_changed_model_settings_observable: Observable<ModelSettingsDTO>;

  constructor(
    private gameStateService: GameStateService
  ) {

    this.gameStateService = gameStateService;
    this.distinct_until_changed_model_settings_observable = this.model_settings_subject
    .pipe(
      distinctUntilChanged((prev, curr) => JSON.stringify(prev) === JSON.stringify(curr))
    );

    merge(
      this.gameStateService.get_observable_state().pipe(map((_value) => null)),
      this.distinct_until_changed_model_settings_observable
      .pipe(
        distinctUntilChanged((prev, curr) => {
          return (
            prev.temperature === curr.temperature &&
            prev.target_type === curr.target_type &&
            prev.prefix === curr.prefix
          );
        }),
        map((_value) => null)
      )
    )
    .pipe(debounceTime(200))
    .subscribe((_) => {
      this.manualRetryTopK();
    })
  }

  public get temperature() {
    return this.model_settings.temperature;
  }
  public set temperature(value) {
    this.model_settings.temperature = value;
    this.model_settings_subject.next(this.model_settings.clone());
  }

  public get doSample() {
    return this.model_settings.do_sample;
  }
  public set doSample(value) {
    this.model_settings.do_sample = value;
    this.model_settings_subject.next(this.model_settings.clone());
  }

  public get commentary_type() {
    return this.model_settings.target_type;
  }
  public set commentary_type(value) {
    this.model_settings.target_type = value;
    this.model_settings_subject.next(this.model_settings.clone());
  }

  public get max_new_tokens() {
    return this.model_settings.max_new_tokens;
  }
  public set max_new_tokens(value) {
    this.model_settings.max_new_tokens = value;
    this.model_settings_subject.next(this.model_settings.clone());
  }

  public get prefix() {
    return this.model_settings.prefix;
  }
  public set prefix(value) {
    this.model_settings.prefix = value;
    this.model_settings_subject.next(this.model_settings.clone());
  }


  getAnnotation(chess: Chess): Observable<string> {
    var current_board = chess.fen();
    var past_boards = chess.history({ verbose: true }).slice(-2).map((move) => move.before);
    var request_dict: AnnotateRequestDict = {
      "past_boards": past_boards,
      "current_board": current_board,
      "do_sample": this.model_settings.do_sample,
      "max_new_tokens": this.model_settings.max_new_tokens,
    };
    if (this.doSample) {
      request_dict["temperature"] = this.model_settings.temperature;
    }
    if (this.commentary_type.length > 0) {
      request_dict["target_type"] = this.model_settings.target_type;
    }
    if (this.prefix.length > 0) {
      request_dict["prefix"] = this.model_settings.prefix.replaceAll("\n", "<n>");
    }
    return fromFetch(environment.modelURL + "/get_commentary", {
      method: "POST",
      body: JSON.stringify(request_dict)
    })
      .pipe(switchMap((result) => {
        if (!result.body) {
          throw "Error reading response";
        }
        return from(result.body! as ReadableStreamLike<Uint8Array>);
      }))
      .pipe(map(bytes => {
        return this.decoder.decode(bytes);
      }));
  }

  getTopK(chess: Chess): Observable<Array<[number, string]>> {
    var current_board = chess.fen();
    var past_boards = chess.history({ verbose: true }).slice(-2).map((move) => move.before);
    var request_dict: TopKRequestDict = {
      "past_boards": past_boards,
      "current_board": current_board,
    };
    if (this.commentary_type.length > 0) {
      request_dict["target_type"] = this.model_settings.target_type;
    }
    if (this.prefix.length > 0) {
      request_dict["prefix"] = this.model_settings.prefix.replaceAll("\n", "<n>");
    }
    request_dict["temperature"] = this.model_settings.temperature;
    return fromFetch(environment.modelURL + "/topk", {
      method: "POST",
      body: JSON.stringify(request_dict)
    })
      .pipe(switchMap(response => {
        return from(response.json());
      }));
  }

  getModelSettingsDistinctUntilChangedObservable(): Observable<ModelSettingsDTO> {
    return this.distinct_until_changed_model_settings_observable;
  }

  getTopKObservable(): Observable<TopKDTO> {
    return this.topk_behavior_subject.asObservable();
  }

  manualRetryTopK(): void {
    this.last_topk_subscription?.unsubscribe();
    this.topk_behavior_subject.next(new TopKDTO([], TopKDTO.State.LOADING));
    this.last_topk_subscription = this.getTopK(this.gameStateService.get_chess_game_at_current_index(2))
    .pipe(retry({ delay: 1000, count: 3}))
    .pipe(map((data) => new TopKDTO(data, TopKDTO.State.LOADED)))
    .subscribe({
      next: (value) => this.topk_behavior_subject.next(value),
      error: () => this.topk_behavior_subject.next(new TopKDTO([], TopKDTO.State.FAILED))
    });
  }
}
