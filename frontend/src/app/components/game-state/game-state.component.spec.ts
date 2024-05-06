import { ComponentFixture, TestBed } from '@angular/core/testing';

import { GameStateComponent } from './game-state.component';
import { Chess } from 'chess.js';
import { BehaviorSubject } from 'rxjs';
import { GameStateService } from '../../services/game-state.service';
import { By } from '@angular/platform-browser';

describe('GameStateComponent', () => {
  let component: GameStateComponent;
  let fixture: ComponentFixture<GameStateComponent>;
  let state_observable: BehaviorSubject<[Chess, number]>;
  let mockGameStateService: jasmine.SpyObj<GameStateService>;
  let chess: Chess;
  let index: number;

  beforeEach(async () => {
    chess = new Chess();
    index = 0;
    state_observable = new BehaviorSubject<[Chess, number]>([chess, index]);
    mockGameStateService = jasmine.createSpyObj('GameStateService', ['get_observable_state', 'get_chess_game_at_current_index', 'get_current_state', 'set_current_fen', 'set_pgn']);
    mockGameStateService.get_observable_state.and.returnValue(state_observable.asObservable());
    mockGameStateService.get_chess_game_at_current_index.and.returnValue(chess);
    mockGameStateService.get_current_state.and.returnValue([chess, index]);

    await TestBed.configureTestingModule({
      imports: [GameStateComponent],
      providers: [ { provide: GameStateService, useValue: mockGameStateService } ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(GameStateComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should update fen/pgn', () => {
    chess.move('e2e4');
    index += 1;
    state_observable.next([chess, index]);
    mockGameStateService.get_chess_game_at_current_index.and.returnValue(chess);

    expect(component.fen).withContext('fen property check after first move').toEqual('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1');
    expect(component.pgn).withContext('pgn property check after first move').toEqual('1. e4');
  });

  it('should mark invalid fen, and change it back on focus out', () => {
    chess.move('e2e4');
    index += 1;
    state_observable.next([chess, index]);
    mockGameStateService.get_chess_game_at_current_index.and.returnValue(chess);

    component.fen = "invalid";
    component.onEnterFEN();

    expect(component.fen_invalid).toBeTrue();
    expect(component.pgn).withContext('pgn should be unchanged').toEqual('1. e4');
    expect(component.fen).withContext('fen should remain invalid until focusout').toEqual('invalid');

    component.onFocusOut();

    expect(component.fen_invalid).toBeFalse();
    expect(component.pgn).withContext('pgn should be unchanged').toEqual('1. e4');
    expect(component.fen).withContext('fen should go back to initial').toEqual('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1');
  });

  it('should update on valid fen', () => {
    chess.move('e2e4');
    index += 1;
    state_observable.next([chess, index]);
    mockGameStateService.get_chess_game_at_current_index.and.returnValue(chess);

    let fen = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1';
    component.fen = fen;

    mockGameStateService.set_current_fen.and.callFake((_): Error | null => {
      chess = new Chess(fen);
      index = 0;
      state_observable.next([chess, index]);
      return null;
    });

    component.onEnterFEN();

    expect(component.fen_invalid).toBeFalse();
    expect(component.pgn).withContext('pgn should contain only metadata').toEqual('[SetUp "1"]\n[FEN "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"]\n');
    expect(component.fen).withContext('fen should be the set fen').toEqual(fen);
  });

  it('should mark invalid pgn, and change it back on focus out', () => {
    chess.move('e2e4');
    index += 1;
    state_observable.next([chess, index]);
    mockGameStateService.get_chess_game_at_current_index.and.returnValue(chess);

    component.pgn = "invalid";

    mockGameStateService.set_pgn.and.callFake((_): Error | null => {
      return new Error("test");
    });

    component.onEnterPGN();

    expect(component.pgn_invalid).toBeTrue();
    expect(component.fen).withContext('fen should be unchanged').toEqual('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1');
    expect(component.pgn).withContext('pgn should remain invalid until focusout').toEqual('invalid');

    component.onFocusOut();

    expect(component.pgn_invalid).toBeFalse();
    expect(component.fen).withContext('fen should be unchanged').toEqual('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1');
    expect(component.pgn).withContext('pgn should go back to initial').toEqual('1. e4');
  });

  it('should update on valid pgn', () => {
    chess.move('e2e4');
    index += 1;
    state_observable.next([chess, index]);
    mockGameStateService.get_chess_game_at_current_index.and.returnValue(chess);

    component.pgn = "1. d4 Nf6";

    mockGameStateService.set_pgn.and.callFake((_): Error | null => {
      chess.reset();
      chess.move('d4');
      chess.move('Nf6');
      index = 0;
      state_observable.next([chess, index]);
      return null;
    });

    component.onEnterPGN();

    expect(component.pgn_invalid).toBeFalse();
    expect(component.fen).withContext('fen should be changed').toEqual('rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2');
    expect(component.pgn).withContext('pgn should be changed').toEqual('1. d4 Nf6');
  });

  it('should mark invalid fen, and change it back on focus out DOM events', () => {
    chess.move('e2e4');
    index += 1;
    state_observable.next([chess, index]);
    mockGameStateService.get_chess_game_at_current_index.and.returnValue(chess);

    let fen_input = fixture.debugElement.query(By.css('#FEN'));

    component.fen = "invalid";
    fen_input.triggerEventHandler('keyup.enter');
    fixture.detectChanges();


    expect(component.fen_invalid).toBeTrue();
    expect(component.pgn).withContext('pgn should be unchanged').toEqual('1. e4');
    expect(component.fen).withContext('fen should remain invalid until focusout').toEqual('invalid');

    fen_input.triggerEventHandler('focusout');
    fixture.detectChanges();

    expect(component.fen_invalid).toBeFalse();
    expect(component.pgn).withContext('pgn should be unchanged').toEqual('1. e4');
    expect(component.fen).withContext('fen should go back to initial').toEqual('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1');
  });

  it('should update on valid fen DOM events', () => {
    chess.move('e2e4');
    index += 1;
    state_observable.next([chess, index]);
    mockGameStateService.get_chess_game_at_current_index.and.returnValue(chess);

    let fen_input = fixture.debugElement.query(By.css('#FEN'));
    let fen = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1';

    component.fen = fen;

    mockGameStateService.set_current_fen.and.callFake((_): Error | null => {
      chess = new Chess(fen);
      index = 0;
      state_observable.next([chess, index]);
      return null;
    });

    fen_input.triggerEventHandler('keyup.enter');
    fixture.detectChanges();

    expect(component.fen_invalid).toBeFalse();
    expect(component.pgn).withContext('pgn should contain only metadata').toEqual('[SetUp "1"]\n[FEN "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"]\n');
    expect(component.fen).withContext('fen should be the set fen').toEqual(fen);
  });

  it('should mark invalid pgn, and change it back on focus out DOM events', () => {
    chess.move('e2e4');
    index += 1;
    state_observable.next([chess, index]);
    mockGameStateService.get_chess_game_at_current_index.and.returnValue(chess);

    let pgn_input = fixture.debugElement.query(By.css("#PGN"));

    component.pgn = "invalid";

    mockGameStateService.set_pgn.and.callFake((_): Error | null => {
      return new Error("test");
    });

    pgn_input.triggerEventHandler('keyup.enter');
    fixture.detectChanges();

    expect(component.pgn_invalid).toBeTrue();
    expect(component.fen).withContext('fen should be unchanged').toEqual('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1');
    expect(component.pgn).withContext('pgn should remain invalid until focusout').toEqual('invalid');

    pgn_input.triggerEventHandler('focusout');
    fixture.detectChanges();

    expect(component.pgn_invalid).toBeFalse();
    expect(component.fen).withContext('fen should be unchanged').toEqual('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1');
    expect(component.pgn).withContext('pgn should go back to initial').toEqual('1. e4');
  });

  it('should update on valid pgn DOM events', () => {
    chess.move('e2e4');
    index += 1;
    state_observable.next([chess, index]);
    mockGameStateService.get_chess_game_at_current_index.and.returnValue(chess);

    let pgn_input = fixture.debugElement.query(By.css("#PGN"));

    component.pgn = "1. d4 Nf6";

    mockGameStateService.set_pgn.and.callFake((_): Error | null => {
      chess.reset();
      chess.move('d4');
      chess.move('Nf6');
      index = 0;
      state_observable.next([chess, index]);
      return null;
    });

    pgn_input.triggerEventHandler('keyup.enter');
    fixture.detectChanges();

    expect(component.pgn_invalid).toBeFalse();
    expect(component.fen).withContext('fen should be changed').toEqual('rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2');
    expect(component.pgn).withContext('pgn should be changed').toEqual('1. d4 Nf6');
  });

});
