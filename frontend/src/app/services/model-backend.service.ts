import { Injectable } from '@angular/core';
import { environment } from '../../environments/environment';
import { GameStateService } from './game-state.service';
import { Chess } from 'chess.js';
import { Observable, map, switchMap, from, BehaviorSubject, ReadableStreamLike, merge, debounceTime, retry, Subscription, distinctUntilChanged, filter, throwError } from 'rxjs';
import { fromFetch } from "rxjs/fetch";
import { TopKDTO } from '../dto/topkDTO';
import { ModelSettingsDTO } from '../dto/modelSettingsDTO';
import { ProgressEnum } from '../enums/ProgressEnum';

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


  private model_settings: ModelSettingsDTO | null = null;
  private decoder = new TextDecoder("utf-8");
  private topk_behavior_subject = new BehaviorSubject<TopKDTO>(new TopKDTO([], ProgressEnum.FAILED));
  private last_topk_subscription: Subscription | null = null;
  private last_model_settings_subscription: Subscription | null = null;

  private model_settings_subject = new BehaviorSubject<ModelSettingsDTO | null>(null);
  private distinct_until_changed_model_settings_observable: Observable<ModelSettingsDTO>;
  private model_settings_progress_behavior_subject = new BehaviorSubject<ProgressEnum>(ProgressEnum.FAILED);

  constructor(
    private gameStateService: GameStateService
  ) {

    this.gameStateService = gameStateService;
    this.distinct_until_changed_model_settings_observable = this.model_settings_subject
    .pipe(
      filter((value: ModelSettingsDTO | null) => value !== null),
      map((value) => value as ModelSettingsDTO),
      distinctUntilChanged((prev, curr) => {
        return JSON.stringify(prev) === JSON.stringify(curr);
      })
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
    this.manualRetryModelSettings();
  }

  public set_temperature(value: number) {
    if(this.model_settings === null) {
      return ;
    }
    this.model_settings.temperature = value;
    this.model_settings_subject.next(this.model_settings.clone());
  }

  public set_doSample(value: boolean){
    if(this.model_settings === null) {
      return ;
    }
    this.model_settings.do_sample = value;
    this.model_settings_subject.next(this.model_settings.clone());
  }

  public set_commentary_type(value: string) {
    if(this.model_settings === null) {
      return ;
    }
    this.model_settings.target_type = value;
    this.model_settings_subject.next(this.model_settings.clone());
  }

  public set_max_new_tokens(value: number) {
    if(this.model_settings === null) {
      return ;
    }
    this.model_settings.max_new_tokens = value;
    this.model_settings_subject.next(this.model_settings.clone());
  }

  public set_prefix(value: string) {
    if(this.model_settings === null) {
      return ;
    }
    this.model_settings.prefix = value;
    this.model_settings_subject.next(this.model_settings.clone());
  }

  public get_prefix(): string {
    return this.model_settings?.prefix || "";
  }

  getAnnotation(chess: Chess): Observable<string> {
    if(this.model_settings === null) {
      return throwError(() => "Model settings have not been initialized yet");
    }
    var current_board = chess.fen();
    let history = chess.history({ verbose: true });
    var past_boards = history.slice(history.length - this.model_settings.count_past_boards).map((move) => move.before);
    var request_dict: AnnotateRequestDict = {
      "past_boards": past_boards,
      "current_board": current_board,
      "do_sample": this.model_settings.do_sample,
      "max_new_tokens": this.model_settings.max_new_tokens,
    };
    if (this.model_settings.do_sample) {
      request_dict["temperature"] = this.model_settings.temperature;
    }
    if (this.model_settings.target_type.length > 0) {
      request_dict["target_type"] = this.model_settings.target_type;
    }
    if (this.model_settings.prefix.length > 0) {
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
    if(this.model_settings === null) {
      return throwError(() => "Model settings have not been initialized yet");
    }
    var current_board = chess.fen();
    let history = chess.history({ verbose: true });
    var past_boards = history.slice(history.length - this.model_settings.count_past_boards).map((move) => move.before);

    var request_dict: TopKRequestDict = {
      "past_boards": past_boards,
      "current_board": current_board,
    };
    if (this.model_settings.target_type.length > 0) {
      request_dict["target_type"] = this.model_settings.target_type;
    }
    if (this.model_settings.prefix.length > 0) {
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

  getInfoObservable(): Observable<ModelSettingsDTO> {
    return fromFetch(environment.modelURL + "/info", { method: "GET" })
    .pipe(
      switchMap(response => from(response.json())),
      map((json) => {
        return new ModelSettingsDTO({
          do_sample: json['do_sample'],
          temperature: json['temperature'],
          min_temperature: json['min_temperature'],
          max_temperature: json['max_temperature'],
          target_type: json['target_type'],
          max_new_tokens: json['max_new_tokens'],
          max_max_new_tokens: json['max_max_new_tokens'],
          prefix: json['prefix'],
          commentary_types: json['commentary_types'],
          count_past_boards: json['count_past_boards']
        });
      })
    );
  }

  getModelSettingsDistinctUntilChangedObservable(): Observable<ModelSettingsDTO> {
    return this.distinct_until_changed_model_settings_observable;
  }

  getTopKObservable(): Observable<TopKDTO> {
    return this.topk_behavior_subject.asObservable();
  }

  getModelSettingsProgressObservable(): Observable<ProgressEnum> {
    return this.model_settings_progress_behavior_subject.asObservable();
  }

  manualRetryTopK(): void {
    this.last_topk_subscription?.unsubscribe();
    this.topk_behavior_subject.next(new TopKDTO([], ProgressEnum.LOADING));
    this.last_topk_subscription = this.getTopK(this.gameStateService.get_chess_game_at_current_index(this.model_settings?.count_past_boards || 0))
    .pipe(retry({ delay: 1000, count: 3}))
    .pipe(map((data) => new TopKDTO(data, ProgressEnum.LOADED)))
    .subscribe({
      next: (value) => this.topk_behavior_subject.next(value),
      error: () => this.topk_behavior_subject.next(new TopKDTO([], ProgressEnum.FAILED))
    });
  }

  manualRetryModelSettings(): void {
    this.last_model_settings_subscription?.unsubscribe();
    this.model_settings_progress_behavior_subject.next(ProgressEnum.LOADING);
    this.last_model_settings_subscription = this.getInfoObservable()
    .pipe(retry({ delay: 1000, count: 3}))
    .subscribe({
      next: (value) => {
        this.model_settings = value;
        this.model_settings_subject.next(value.clone());
        this.model_settings_progress_behavior_subject.next(ProgressEnum.LOADED);
      },
      error: () => this.model_settings_progress_behavior_subject.next(ProgressEnum.FAILED)
    });
  }

  retryAll(): void {
    if(this.model_settings === null) {
      this.manualRetryModelSettings();
    }
    this.manualRetryTopK();
  }
}
