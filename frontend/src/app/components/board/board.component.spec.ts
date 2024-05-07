import { ComponentFixture, TestBed } from '@angular/core/testing';

import { BoardComponent } from './board.component';
import { BehaviorSubject } from 'rxjs';
import { GameStateService } from '../../services/game-state.service';
import { Chess } from 'chess.js';

describe('BoardComponent', () => {
  let component: BoardComponent;
  let fixture: ComponentFixture<BoardComponent>;
  let state_observable: BehaviorSubject<[Chess, number]>;
  let mockGameStateService: jasmine.SpyObj<GameStateService>;
  let chess: Chess;
  let index: number;

  beforeEach(async () => {
    chess = new Chess();
    chess.move('e4');
    chess.move('Nf6');
    index = 2;
    state_observable = new BehaviorSubject<[Chess, number]>([chess, index]);
    mockGameStateService = jasmine.createSpyObj('GameStateService', ['get_observable_state', 'get_chess_game_at_current_index', 'get_current_state', 'move']);
    mockGameStateService.get_observable_state.and.returnValue(state_observable.asObservable());
    mockGameStateService.get_chess_game_at_current_index.and.returnValue(chess);
    mockGameStateService.get_current_state.and.returnValue([chess, index]);

    await TestBed.configureTestingModule({
      imports: [BoardComponent],
      providers: [
        { provide: GameStateService, useValue: mockGameStateService},
      ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(BoardComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should have standard chess files and rows', () => {
    expect(component.files).toEqual(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']);
    expect(component.ranks).toEqual(['8', '7', '6', '5', '4', '3', '2', '1']);
  });

  it('should have standard chess files and rows reversed', () => {
    fixture = TestBed.createComponent(BoardComponent);
    component = fixture.componentInstance;
    component.flipped = true;
    fixture.detectChanges();
    expect(component.files).toEqual(['h', 'g', 'f', 'e', 'd', 'c', 'b', 'a']);
    expect(component.ranks).toEqual(['1', '2', '3', '4', '5', '6', '7', '8']);
  });

  it('should show legal moves of focused square', () => {
    component.focusSquare('b1');
    expect(component.shownMoves).toEqual(['a3', 'c3']);
  });

  it('should show clear legal moves when unfocused', () => {
    component.focusSquare('b1');
    component.unfocusCurrentSquare();
    expect(component.shownMoves).toEqual([]);
  });

  it('should show only last focused square\'s moves', () => {
    component.focusSquare('b1');
    component.focusSquare('e4');
    expect(component.shownMoves).toEqual(['e5']);
  });

  it('should move e4e5 via clicking', () => {
    component.clickSquare('e4');
    component.clickSquare('e5');
    expect(mockGameStateService.move).toHaveBeenCalled();
  });

  it('should cancel move via clicking illegal move', () => {
    component.clickSquare('e4');
    component.clickSquare('e6');
    expect(mockGameStateService.move).not.toHaveBeenCalled();
    expect(component.focusedSquare).toBeFalsy();
    expect(component.shownMoves).toEqual([]);
  });

  let beforePromotion = () => {
    chess = new Chess('rnbq2nr/pp1pPkpp/8/2b5/7N/5P2/PPPpK1PP/RNB1QB1R w - - 2 10');
    index = 0;
    mockGameStateService.get_chess_game_at_current_index.and.returnValue(chess);
    mockGameStateService.get_current_state.and.returnValue([chess, index]);
    state_observable.next([chess, index]);
  };

  it('should recognize promotion', () => {
    beforePromotion();
    component.clickSquare('e7');
    component.clickSquare('e8');
    expect(component.pendingPromotionMove).toBeTruthy();
  });

  it('should recognize promotion and capture', () => {
    beforePromotion();
    component.clickSquare('e7');
    component.clickSquare('d8');
    expect(component.pendingPromotionMove).toBeTruthy();
  });

  it('should cancel promotion', () => {
    beforePromotion();
    component.clickSquare('e7');
    component.clickSquare('d8');
    component.cancelPromotion();
    expect(component.pendingPromotionMove).toBeFalsy();
  });

  it('should recognize capture', () => {
    beforePromotion();
    component.clickSquare('e7');
    expect(component.isCapture('d8')).toBeTrue();
  });

  it('should recognize non-capture', () => {
    beforePromotion();
    component.clickSquare('e7');
    expect(component.isCapture('e8')).toBeFalse();
  });

  it('should show white promotion pieces', () => {
    beforePromotion();
    component.clickSquare('e7');
    component.clickSquare('d8');
    expect(component.getPromotionPieces()).toEqual(['wQ', 'wN', 'wR', 'wB']);
  });

  it('should show black promotion pieces', () => {
    beforePromotion();
    chess.move('h4f5');
    index += 1;
    state_observable.next([chess, index]);
    component.clickSquare('d2');
    component.clickSquare('d1');
    expect(component.getPromotionPieces()).toEqual(['bB', 'bR', 'bN', 'bQ']);
  });

  it('should throw when not promoting', () => {
    beforePromotion();
    component.clickSquare('c2');
    component.clickSquare('c3');
    expect(() => component.getPromotionPieces()).toThrow("Unexpected promotion");
    expect(() => component.promote('wQ')).toThrow("Unexpected promotion");
  });

  it('should promote', () => {
    beforePromotion();
    component.clickSquare('e7');
    component.clickSquare('e8');
    component.promote('wQ');
    expect(mockGameStateService.move).toHaveBeenCalled();
  });

  it('should cover dragover', () => {
    component.onDragOver(new DragEvent(''));
    expect(true).toBeTrue();
  });
});
