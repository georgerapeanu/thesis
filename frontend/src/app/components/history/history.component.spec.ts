import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HistoryComponent } from './history.component';
import { GameStateService } from '../../services/game-state.service';
import { ChessEngineService } from '../../services/chess-engine.service';
import { Chess } from 'chess.js';
import { BehaviorSubject, Subject } from 'rxjs';
import { EvaluationDTO } from '../../dto/evaluationDTO';

describe('HistoryComponent', () => {
  let component: HistoryComponent;
  let fixture: ComponentFixture<HistoryComponent>;
  let gameStateService: jasmine.SpyObj<GameStateService>;
  let chessEngineService: jasmine.SpyObj<ChessEngineService>;
  let chess: Chess;
  let index: number;
  let gameStateObservable: BehaviorSubject<[Chess, number]>;
  let chessEvaluationObservable: Subject<EvaluationDTO>;

  beforeEach(async () => {
    gameStateService = jasmine.createSpyObj<GameStateService>('GameStateService', [
      'get_observable_state',
      'get_chess_game_at_current_index',
      'get_current_state',
      'seek',
      'undo',
      'redo'
    ]);
    chessEngineService = jasmine.createSpyObj<ChessEngineService>('ChessEngineService', [
      'requestEvaluation'
    ]);

    chess = new Chess();
    chess.move('e4');
    chess.move('Nf6');
    index = 2;

    gameStateObservable = new BehaviorSubject<[Chess, number]>([chess, index]);
    chessEvaluationObservable = new Subject<EvaluationDTO>();

    chessEngineService.requestEvaluation.and.returnValue(chessEvaluationObservable.asObservable());
    gameStateService.get_observable_state.and.returnValue(gameStateObservable.asObservable());
    gameStateService.get_chess_game_at_current_index.and.returnValue(chess);
    gameStateService.get_current_state.and.returnValue([chess, index]);



    await TestBed.configureTestingModule({
      imports: [HistoryComponent],
      providers: [
        { provide: GameStateService, useValue: gameStateService},
        { provide: ChessEngineService, useValue: chessEngineService},
      ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(HistoryComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should request evaluation and update values accordingly', () => {
    expect(component.evaluationPending).toBeTrue();
    chessEvaluationObservable.next(new EvaluationDTO(false, 1, 1));
    expect(component.evaluationPending).toBeFalse();
  });

  it('should undo on DOM ArrowLeft event', (done) => {
    let event = new KeyboardEvent("keydown", {
      "key": "ArrowLeft"
    });
    const divElement = document.createElement('div');
    spyOnProperty(event, 'target').and.returnValue(divElement);

    document.dispatchEvent(event);
    fixture.detectChanges();
    setTimeout(() => {
      expect(gameStateService.undo).toHaveBeenCalled();
      done();
    }, 100);
  });

  it('should redo on DOM ArrowRight event', (done) => {
    let event = new KeyboardEvent("keydown", {
      "key": "ArrowRight"
    });
    const divElement = document.createElement('div');
    spyOnProperty(event, 'target').and.returnValue(divElement);

    document.dispatchEvent(event);
    fixture.detectChanges();
    setTimeout(() => {
      expect(gameStateService.redo).toHaveBeenCalled();
      done();
    }, 100);
  });

  it('should seek beginning on DOM ArrowUp event', (done) => {
    let event = new KeyboardEvent("keydown", {
      "key": "ArrowUp"
    });
    const divElement = document.createElement('div');
    spyOnProperty(event, 'target').and.returnValue(divElement);

    document.dispatchEvent(event);
    fixture.detectChanges();
    setTimeout(() => {
      expect(gameStateService.seek).toHaveBeenCalledWith(0);
      done();
    }, 100);
  });

  it('should seek beginning on DOM ArrowDown event', (done) => {
    let event = new KeyboardEvent("keydown", {
      "key": "ArrowDown"
    });
    const divElement = document.createElement('div');
    spyOnProperty(event, 'target').and.returnValue(divElement);

    document.dispatchEvent(event);
    fixture.detectChanges();
    setTimeout(() => {
      expect(gameStateService.seek).toHaveBeenCalledWith(2);
      done();
    }, 100);
  });

  it('should ignore events targeted to inputs', (done) => {
    let event = new KeyboardEvent("keydown", {
      "key": "ArrowDown"
    });
    const inputElement = document.createElement('input');
    spyOnProperty(event, 'target').and.returnValue(inputElement);

    document.dispatchEvent(event);
    fixture.detectChanges();
    setTimeout(() => {
      expect(gameStateService.seek).not.toHaveBeenCalled();
      done();
    }, 100);
  });

  it('should ignore events targeted to textareas', (done) => {
    let event = new KeyboardEvent("keydown", {
      "key": "ArrowDown"
    });
    const textareaElement = document.createElement('textarea');
    spyOnProperty(event, 'target').and.returnValue(textareaElement);

    document.dispatchEvent(event);
    fixture.detectChanges();
    setTimeout(() => {
      expect(gameStateService.seek).not.toHaveBeenCalled();
      done();
    }, 100);
  });

  it('should ignore events targeted to non html elements', (done) => {
    let event = new KeyboardEvent("keydown", {
      "key": "ArrowDown"
    });
    spyOnProperty(event, 'target').and.returnValue(document);

    document.dispatchEvent(event);
    fixture.detectChanges();
    setTimeout(() => {
      expect(gameStateService.seek).not.toHaveBeenCalled();
      done();
    }, 100);
  });

  it('should ignore null index onclick', () => {
    component.onClick(null);
    expect(gameStateService.seek).not.toHaveBeenCalled();
  });

  it('should focus on cliclked elements', () => {
    component.onClick(1);
    expect(gameStateService.seek).toHaveBeenCalled();
  });

  it('should emit null event on flip request', () => {
    let emmitted = false;
    component.requestFlip.subscribe((value) => {
      expect(value).toBeNull();
      emmitted = true;
    });
    component.onRequestFlip();
    expect(emmitted).toBeTrue();
  });

  it('should unsubscribe from all observables', () => {
    fixture.destroy();
    expect(gameStateObservable.observed).toBeFalse();
    expect(chessEvaluationObservable.observed).toBeFalse();
    expect(component.keyCommandObservable.observed).toBeFalse();
  });
});
