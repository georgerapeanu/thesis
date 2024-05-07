import { TestBed } from '@angular/core/testing';

import { ModelBackendService } from './model-backend.service';
import { HttpClientModule } from '@angular/common/http';
import { BehaviorSubject, Subject, first, firstValueFrom } from 'rxjs';
import { environment } from '../../environments/environment';
import { ModelSettingsDTO } from '../dto/modelSettingsDTO';
import { Chess } from 'chess.js';
import { GameStateService } from './game-state.service';
import { ProgressEnum } from '../enums/ProgressEnum';

function createStringStreamResponseWithExtra(data: string, status: number = 200): Response {
  const encoder = new TextEncoder();
  // Create a ReadableStream from the data string
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode("Chunk "));
      controller.enqueue(encoder.encode(data));
      controller.close();
    }
  });

  // Create headers for the response
  const headers = new Headers();
  headers.set('Content-Type', 'text/plain');

  // Create the response object with the string stream
  const response = new Response(stream, {
    status: status,
    headers: headers
  });

  return response;
}

describe('ModelBackendService', () => {
  let service: ModelBackendService;
  let commentaryObservable: Subject<Response>;
  let infoObservable: Subject<Response>;
  let topkObservable: Subject<Response>;
  let model_settings: ModelSettingsDTO;
  let state_observable: BehaviorSubject<[Chess, number]>;
  let mockGameStateService: jasmine.SpyObj<GameStateService>;
  let chess: Chess;
  let index: number;
  let last_args: any;

  beforeEach(() => {
    model_settings = new ModelSettingsDTO({
      temperature: 2.3,
      do_sample: false,
      commentary_types: [['type 1', 'type 1 actual']],
      target_type: '',
      max_new_tokens: 23,
      max_max_new_tokens: 24,
      min_temperature: 2,
      max_temperature: 10,
      prefix: '2323',
      count_past_boards: 5
    });
    commentaryObservable = new Subject<Response>();
    infoObservable = new Subject<Response>();
    topkObservable = new Subject<Response>();
    chess = new Chess();
    chess.move('e4');
    chess.move('Nf6');
    index = 2;
    state_observable = new BehaviorSubject<[Chess, number]>([chess, index]);
    mockGameStateService = jasmine.createSpyObj('GameStateService', ['get_observable_state', 'get_chess_game_at_current_index', 'get_current_state', 'move']);
    mockGameStateService.get_observable_state.and.returnValue(state_observable.asObservable());
    mockGameStateService.get_chess_game_at_current_index.and.returnValue(chess);
    mockGameStateService.get_current_state.and.returnValue([chess, index]);

    spyOn(window, "fetch")
      .and.callFake((url, args) => {
        last_args = args;
        if(url.toString() === environment.modelURL + "/get_commentary") {
          return (firstValueFrom(commentaryObservable.asObservable()));
        } else if(url.toString() === environment.modelURL + "/topk") {
          return (firstValueFrom(topkObservable.asObservable()));
        } else if(url.toString() === environment.modelURL + "/info") {
          return (firstValueFrom(infoObservable.asObservable()));
        }
        expect(true).toBeFalse();
        return Promise.reject();
      });
    TestBed.configureTestingModule({
      imports: [HttpClientModule],
      providers: [{ provide: GameStateService, useValue: mockGameStateService}]
    });
    service = TestBed.inject(ModelBackendService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should set temperature after receiving model settings', (done) => {
    let model_settings_copy = model_settings.clone();
    let response = new Response(JSON.stringify(model_settings), {
      status: 200,
      headers: {
        'Content-Type': 'application/json'
      }
    });
    infoObservable.next(response);
    service.getModelSettingsDistinctUntilChangedObservable().pipe(first()).subscribe((model_settings) => {
      expect(model_settings).toEqual(model_settings_copy);
      model_settings_copy.temperature = 2.5;
      service.set_temperature(2.5);
      service.getModelSettingsDistinctUntilChangedObservable().pipe(first()).subscribe((model_settings) => {
        expect(model_settings).toEqual(model_settings_copy);
        done();
      });
    });
  });

  it('should set doSample after receiving model settings', (done) => {
    let model_settings_copy = model_settings.clone();
    let response = new Response(JSON.stringify(model_settings), {
      status: 200,
      headers: {
        'Content-Type': 'application/json'
      }
    });
    infoObservable.next(response);
    service.getModelSettingsDistinctUntilChangedObservable().pipe(first()).subscribe((model_settings) => {
      expect(model_settings).toEqual(model_settings_copy);
      model_settings_copy.do_sample = true;
      service.set_doSample(true);
      service.getModelSettingsDistinctUntilChangedObservable().pipe(first()).subscribe((model_settings) => {
        expect(model_settings).toEqual(model_settings_copy);
        done();
      });
    });
  });

  it('should set commentary_type after receiving model settings', (done) => {
    let model_settings_copy = model_settings.clone();
    let response = new Response(JSON.stringify(model_settings), {
      status: 200,
      headers: {
        'Content-Type': 'application/json'
      }
    });
    infoObservable.next(response);
    service.getModelSettingsDistinctUntilChangedObservable().pipe(first()).subscribe((model_settings) => {
      expect(model_settings).toEqual(model_settings_copy);
      model_settings_copy.target_type = 'type 1 actual';
      service.set_commentary_type('type 1 actual');
      service.getModelSettingsDistinctUntilChangedObservable().pipe(first()).subscribe((model_settings) => {
        expect(model_settings).toEqual(model_settings_copy);
        done();
      });
    });
  });

  it('should set max_new_tokens after receiving model settings', (done) => {
    let model_settings_copy = model_settings.clone();
    let response = new Response(JSON.stringify(model_settings), {
      status: 200,
      headers: {
        'Content-Type': 'application/json'
      }
    });
    infoObservable.next(response);
    service.getModelSettingsDistinctUntilChangedObservable().pipe(first()).subscribe((model_settings) => {
      expect(model_settings).toEqual(model_settings_copy);
      model_settings_copy.max_new_tokens = 99;
      service.set_max_new_tokens(99);
      service.getModelSettingsDistinctUntilChangedObservable().pipe(first()).subscribe((model_settings) => {
        expect(model_settings).toEqual(model_settings_copy);
        done();
      });
    });
  });

  it('should set prefix after receiving model settings', (done) => {
    let model_settings_copy = model_settings.clone();
    let response = new Response(JSON.stringify(model_settings), {
      status: 200,
      headers: {
        'Content-Type': 'application/json'
      }
    });
    infoObservable.next(response);
    service.getModelSettingsDistinctUntilChangedObservable().pipe(first()).subscribe((model_settings) => {
      expect(model_settings).toEqual(model_settings_copy);
      model_settings_copy.prefix = '99';
      service.set_prefix('99');
      service.getModelSettingsDistinctUntilChangedObservable().pipe(first()).subscribe((model_settings) => {
        expect(model_settings).toEqual(model_settings_copy);
        done();
      });
    });
  });

  it('should get prefix after receiving model settings', (done) => {
    let response = new Response(JSON.stringify(model_settings), {
      status: 200,
      headers: {
        'Content-Type': 'application/json'
      }
    });
    infoObservable.next(response);
    service.getModelSettingsDistinctUntilChangedObservable().pipe(first()).subscribe((model_settings) => {
      expect(model_settings.prefix).toEqual('2323');
      done();
    });
  });

  it('should ignore setters if it didn\' receive settings', () => {
    let called = false;
    service.getModelSettingsDistinctUntilChangedObservable().pipe().subscribe((_model_settings) => {
      called = true;
    });
    service.set_prefix("asfas");
    service.set_max_new_tokens(12);
    service.set_commentary_type("asfas");
    service.set_doSample(true);
    service.set_temperature(244);
    expect(called).toBeFalse();
  });

  it('should return empty prefix if not initialized', () => {
    expect(service.get_prefix()).toEqual('');
  });

  it('should return annotation with correct payload vary boards', (done) => {
    model_settings.count_past_boards = 1;
    let response = new Response(JSON.stringify(model_settings), {
      status: 200,
      headers: {
        'Content-Type': 'application/json'
      }
    });
    infoObservable.next(response);
    service.getModelSettingsDistinctUntilChangedObservable().pipe(first()).subscribe((_model_settings) => {
      let prefix = "";
      service.getAnnotation(chess).subscribe({
        next: (value) => {
          prefix += value;
        },
        complete: () => {
          expect(prefix).toEqual("Chunk This is a test");
          expect(JSON.parse(last_args.body)).toEqual({
            "past_boards":["rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"],
            "current_board":"rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
            "do_sample":false,
            "max_new_tokens":23,
            "prefix":"2323"
          });
          done();
        }
      });
      let response = createStringStreamResponseWithExtra("This is a test");
      commentaryObservable.next(response);
    });
  });

  it('should return annotation with correct payload', (done) => {
    model_settings.count_past_boards = 2;
    let response = new Response(JSON.stringify(model_settings), {
      status: 200,
      headers: {
        'Content-Type': 'application/json'
      }
    });
    infoObservable.next(response);
    service.getModelSettingsDistinctUntilChangedObservable().pipe(first()).subscribe((_model_settings) => {
      let prefix = "";
      service.getAnnotation(chess).subscribe({
        next: (value) => {
          prefix += value;
        },
        complete: () => {
          expect(prefix).toEqual("Chunk This is a test");
          expect(JSON.parse(last_args.body)).toEqual({
            "past_boards": ['rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1'],
            "current_board":"rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
            "do_sample":false,
            "max_new_tokens":23,
            "prefix":"2323"
          });
          done();
        }
      });
      let response = createStringStreamResponseWithExtra("This is a test");
      commentaryObservable.next(response);
    });
  });

  it('should return annotation with correct payload with do_sample', (done) => {
    model_settings.do_sample = true;
    let response = new Response(JSON.stringify(model_settings), {
      status: 200,
      headers: {
        'Content-Type': 'application/json'
      }
    });
    infoObservable.next(response);
    service.getModelSettingsDistinctUntilChangedObservable().pipe(first()).subscribe((_model_settings) => {
      let prefix = "";
      service.getAnnotation(chess).subscribe({
        next: (value) => {
          prefix += value;
        },
        complete: () => {
          expect(prefix).toEqual("Chunk This is a test");
          expect(JSON.parse(last_args.body)).toEqual({
            "past_boards": ['rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1'],
            "current_board":"rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
            "do_sample":true,
            "max_new_tokens":23,
            "prefix":"2323",
            "temperature": 2.3
          });
          done();
        }
      });
      let response = createStringStreamResponseWithExtra("This is a test");
      commentaryObservable.next(response);
    });
  });

  it('should return annotation with correct payload with empty prefix', (done) => {
    model_settings.prefix = "";
    let response = new Response(JSON.stringify(model_settings), {
      status: 200,
      headers: {
        'Content-Type': 'application/json'
      }
    });
    infoObservable.next(response);
    service.getModelSettingsDistinctUntilChangedObservable().pipe(first()).subscribe((_model_settings) => {
      let prefix = "";
      service.getAnnotation(chess).subscribe({
        next: (value) => {
          prefix += value;
        },
        complete: () => {
          expect(prefix).toEqual("Chunk This is a test");
          expect(JSON.parse(last_args.body)).toEqual({
            "past_boards": ['rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1'],
            "current_board":"rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
            "do_sample":false,
            "max_new_tokens":23
          });
          done();
        }
      });
      let response = createStringStreamResponseWithExtra("This is a test");
      commentaryObservable.next(response);
    });
  });

  it('should return annotation with correct payload with target type', (done) => {
    model_settings.target_type = "asfas";
    let response = new Response(JSON.stringify(model_settings), {
      status: 200,
      headers: {
        'Content-Type': 'application/json'
      }
    });
    infoObservable.next(response);
    service.getModelSettingsDistinctUntilChangedObservable().pipe(first()).subscribe((_model_settings) => {
      let prefix = "";
      service.getAnnotation(chess).subscribe({
        next: (value) => {
          prefix += value;
        },
        complete: () => {
          expect(prefix).toEqual("Chunk This is a test");
          expect(JSON.parse(last_args.body)).toEqual({
            "past_boards": ['rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1'],
            "current_board":"rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
            "do_sample":false,
            "max_new_tokens":23,
            "prefix":"2323",
            "target_type": "asfas"
          });
          done();
        }
      });
      let response = createStringStreamResponseWithExtra("This is a test");
      commentaryObservable.next(response);
    });
  });

  it('should error annotation if no settings are setup yet', (done) => {
    service.getAnnotation(chess).subscribe({
      error: (_value) => {
        expect(true).toBeTrue();
        done();
      },
    });
    let response = createStringStreamResponseWithExtra("This is a test");
    commentaryObservable.next(response);
  });

  it('should error annotation on incorrect response', (done) => {
    model_settings.target_type = "asfas";
    let response = new Response(JSON.stringify(model_settings), {
      status: 200,
      headers: {
        'Content-Type': 'application/json'
      }
    });
    infoObservable.next(response);
    service.getModelSettingsDistinctUntilChangedObservable().pipe(first()).subscribe((_model_settings) => {
      service.getAnnotation(chess).subscribe({
        error: (_value) => {
          expect(true).toBeTrue();
          done();
        },
      });
      let response = new Response();
      commentaryObservable.next(response);
    });
  });

  it('should return topk with correct payload vary boards', (done) => {
    model_settings.count_past_boards = 1;
    let response = new Response(JSON.stringify(model_settings), {
      status: 200,
      headers: {
        'Content-Type': 'application/json'
      }
    });
    infoObservable.next(response);
    service.getModelSettingsDistinctUntilChangedObservable().pipe(first()).subscribe((_model_settings) => {
      service.retryAll();
      service.getTopKObservable().pipe(first()).subscribe({
        next: (_value) => {
          expect(JSON.parse(last_args.body)).toEqual({
            "past_boards":["rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"],
            "current_board":"rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
            "prefix":"2323",
            "temperature": 2.3
          });
          done();
        },
      });
      topkObservable.next(new Response(JSON.stringify([
        [0.3, "a"], [0.7, "b"]
      ]), {
        status:200,
        headers: {
          'Content-Type': 'application/json'
        }
      }));
    });
  });

  it('should emit failed progress when model settings is not able to be fetched', (done) => {
    service.getModelSettingsProgressObservable().subscribe({
      next: (progress) => {
        if(progress != ProgressEnum.LOADING) {
          expect(progress).toEqual(ProgressEnum.FAILED);
          done();
        }
      }
    });
    infoObservable.error("[]");
    infoObservable.error("[]");
    infoObservable.error("[]");
    infoObservable.error("[]");
  });
});
