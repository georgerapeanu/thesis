import { TestBed } from '@angular/core/testing';

import { ChessEngineService } from './chess-engine.service';
import { Chess } from 'chess.js';
import { first } from 'rxjs';

describe('ChessEngineService', () => {
  let service: ChessEngineService;
  let chess: Chess;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(ChessEngineService);
    chess = new Chess();
    chess.move('e4');
    chess.move('Nf6');
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should provide evaluation observable', (done) => {
    let observable = service.requestEvaluation(chess);
    observable.pipe(first()).subscribe((value) => {
      expect(value).toBeTruthy();
      done();
    });
  });

  it('should complete previous evaluation before starting new one', (done) => {
    let observable = service.requestEvaluation(chess);
    let completed = false;
    observable.subscribe({
      complete: () => {
        completed = true;
      }
    });

    chess.undo();
    let second_observable = service.requestEvaluation(chess);
    second_observable.pipe(first()).subscribe((_) => {
      expect(completed).toBeTrue();
      done();
    });
  });

  it('should return null evaluations when stockfish is not available', (done) => {
    service.stockfish = null;
    let observable = service.requestEvaluation(chess);
    observable.pipe(first()).subscribe((value) => {
      expect(value).toBeNull();
      done();
    });
  });

  it('should resort to js stockfish if wasm is not available', (done) => {
    spyOn(WebAssembly, 'validate').and.returnValue(false);
    service = new ChessEngineService();
    let observable = service.requestEvaluation(chess);
    observable.pipe(first()).subscribe((value) => {
      expect(value).toBeTruthy();
      done();
    });
  });
});
