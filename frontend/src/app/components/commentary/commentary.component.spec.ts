import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CommentaryComponent } from './commentary.component';
import { ModelBackendService } from '../../services/model-backend.service';
import { Chess } from 'chess.js';
import { GameStateService } from '../../services/game-state.service';
import { BehaviorSubject, Subject } from 'rxjs';
import { By } from '@angular/platform-browser';
import { ModelSettingsDTO } from '../../dto/modelSettingsDTO';

describe('CommentaryComponent', () => {
  let component: CommentaryComponent;
  let fixture: ComponentFixture<CommentaryComponent>;
  let modelBackendService: jasmine.SpyObj<ModelBackendService>;
  let gameStateService: jasmine.SpyObj<GameStateService>;
  let commentary_observable: Subject<string>;
  let model_settings_observable: BehaviorSubject<ModelSettingsDTO>;
  let model_settings: ModelSettingsDTO;


  beforeEach(async () => {
    gameStateService = jasmine.createSpyObj('GameStateService', ['get_chess_game_at_current_index']);

    let chess = new Chess();
    chess.loadPgn('1. e4 Nf6');
    gameStateService.get_chess_game_at_current_index.and.returnValue(chess);

    model_settings = (new ModelSettingsDTO({
      'temperature': 1,
      'max_temperature': 0.1,
      'min_temperature': 3,
      'prefix': "",
      'do_sample': false,
      'commentary_types': [],
      'max_max_new_tokens': 100,
      'max_new_tokens': 100,
      'target_type': "",
      "count_past_boards": 4
    }));
    commentary_observable = new Subject<string>();
    model_settings_observable = new BehaviorSubject(model_settings);
    modelBackendService = jasmine.createSpyObj('ModelBackendService', ['getAnnotation', 'getModelSettingsDistinctUntilChangedObservable']);
    modelBackendService.getAnnotation.and.returnValue(commentary_observable.asObservable());
    modelBackendService.getModelSettingsDistinctUntilChangedObservable.and.returnValue(model_settings_observable.asObservable());

    await TestBed.configureTestingModule({
      imports: [CommentaryComponent],
      providers: [
        { provide: GameStateService, useValue: gameStateService},
        { provide: ModelBackendService, useValue: modelBackendService},
      ]
    })
    .compileComponents();


    fixture = TestBed.createComponent(CommentaryComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should get commentary', () => {
    component.request_commentary();
    commentary_observable.next('A');
    commentary_observable.next(' comm');
    commentary_observable.next('entary');
    commentary_observable.next(' test.');
    commentary_observable.complete();

    expect(component.commentary).toEqual("A commentary test.");
    expect(component.is_placeholder).toBeFalse();
  });

  it('should get commentary DOM', () => {
    fixture.debugElement.query(By.css('#request_commentary_button')).triggerEventHandler('click');

    commentary_observable.next('A');
    commentary_observable.next(' comm');
    commentary_observable.next('entary');
    commentary_observable.next(' test.');
    commentary_observable.complete();

    expect(component.commentary).toEqual("A commentary test.");
    expect(component.is_placeholder).toBeFalse();
  });

  it('should display error on error', () => {
    component.request_commentary();
    commentary_observable.next('aaa');
    commentary_observable.error('');

    expect(component.error).toEqual("The request has failed.");
    expect(component.is_placeholder).toBeFalse();
  });

  it('should unsubscribe from observables', () => {
    component.request_commentary();
    component.request_commentary();
    component.request_commentary();
    component.request_commentary();

    fixture.destroy();
    expect(commentary_observable.observed).toBeFalse();
    expect(model_settings_observable.observed).toBeFalse();
  });
});
