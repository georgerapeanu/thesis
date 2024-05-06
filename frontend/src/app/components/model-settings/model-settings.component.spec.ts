import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ModelSettingsComponent } from './model-settings.component';
import { BehaviorSubject, Subject } from 'rxjs';
import { ModelSettingsDTO } from '../../dto/modelSettingsDTO';
import { ModelBackendService } from '../../services/model-backend.service';
import { By } from '@angular/platform-browser';
import { ProgressEnum } from '../../enums/ProgressEnum';
import { DebugElement } from '@angular/core';

describe('ModelSettingsComponent', () => {
  let component: ModelSettingsComponent;
  let fixture: ComponentFixture<ModelSettingsComponent>;
  let commentaryTypeField: DebugElement;
  let sampleField: DebugElement;
  let temperatureTypeField: DebugElement;
  let maxTokensField: DebugElement;
  let prefixField: DebugElement;

  let model_settings_observable: Subject<ModelSettingsDTO>;
  let model_settings_progress_observable: BehaviorSubject<ProgressEnum>;
  let model_settings: ModelSettingsDTO;
  let modelBackendService: jasmine.SpyObj<ModelBackendService>;

  beforeEach(async () => {


    model_settings = (new ModelSettingsDTO({
      'temperature': 1,
      'max_temperature': 0.1,
      'min_temperature': 3,
      'prefix': "",
      'do_sample': false,
      'commentary_types': [['type 1', 'type 1'], ['type 2', 'type 2']],
      'max_max_new_tokens': 100,
      'max_new_tokens': 100,
      'target_type': ""
    }));
    model_settings_observable = new Subject<ModelSettingsDTO>();
    model_settings_progress_observable = new BehaviorSubject<ProgressEnum>(ProgressEnum.LOADED);
    modelBackendService = jasmine.createSpyObj('ModelBackendService', [
      'getModelSettingsDistinctUntilChangedObservable',
      'getModelSettingsProgressObservable',
      'set_commentary_type',
      'set_doSample',
      'set_temperature',
      'set_max_new_tokens',
      'set_prefix',
      'retryAll'
    ]);
    modelBackendService.getModelSettingsDistinctUntilChangedObservable.and.returnValue(model_settings_observable.asObservable());
    modelBackendService.getModelSettingsProgressObservable.and.returnValue(model_settings_progress_observable.asObservable());

    await TestBed.configureTestingModule({
      imports: [ModelSettingsComponent],
      providers: [
        { provide: ModelBackendService, useValue: modelBackendService},
      ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ModelSettingsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  let beforeOnlineMode = () => {
    model_settings_observable.next(model_settings);
    model_settings_progress_observable.next(ProgressEnum.LOADED);
    fixture.detectChanges();
    commentaryTypeField = fixture.debugElement.query(By.css('#commentary_type'));
    sampleField = fixture.debugElement.query(By.css('#sample'));
    temperatureTypeField = fixture.debugElement.query(By.css('#temperature'));
    maxTokensField = fixture.debugElement.query(By.css('#max_new_tokens'));
    prefixField = fixture.debugElement.query(By.css('#prefix'));
  }

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should update commentary type', () => {
    beforeOnlineMode();
    component.modelSettings!.target_type = "type 1";
    component.updateCommentaryType();
    expect(modelBackendService.set_commentary_type).toHaveBeenCalledWith("type 1");
  });

  it('should update sample', () => {
    beforeOnlineMode();
    component.modelSettings!.do_sample = true;
    component.updateSample();
    expect(modelBackendService.set_doSample).toHaveBeenCalledWith(true);
  });

  it('should update temperature', () => {
    beforeOnlineMode();
    component.modelSettings!.temperature = 2;
    component.updateTemperature();
    expect(modelBackendService.set_temperature).toHaveBeenCalledWith(2);
  });

  it('should update max new tokens', () => {
    beforeOnlineMode();
    component.modelSettings!.max_new_tokens = 9;
    component.updateMaxNewTokens();
    expect(modelBackendService.set_max_new_tokens).toHaveBeenCalledWith(9);
  });

  it('should update prefix', () => {
    beforeOnlineMode();
    component.modelSettings!.prefix = "asdas";
    component.updatePrefix();
    expect(modelBackendService.set_prefix).toHaveBeenCalledWith("asdas");
  });

  it('should not update anything when offline/loading', () => {
    component.updateCommentaryType();
    component.updateSample();
    component.updateTemperature();
    component.updateMaxNewTokens();
    component.updatePrefix();
    expect(modelBackendService.set_commentary_type).not.toHaveBeenCalled();
    expect(modelBackendService.set_doSample).not.toHaveBeenCalled();
    expect(modelBackendService.set_temperature).not.toHaveBeenCalled();
    expect(modelBackendService.set_max_new_tokens).not.toHaveBeenCalled();
    expect(modelBackendService.set_prefix).not.toHaveBeenCalled();
  });

  it('should retry in offline mode', () => {
    model_settings_progress_observable.next(ProgressEnum.FAILED);
    fixture.detectChanges();
    component.onRetryModelSettings();
    expect(modelBackendService.retryAll).toHaveBeenCalled();
  });

  it('should retry in offline mode DOM', () => {
    model_settings_progress_observable.next(ProgressEnum.FAILED);
    fixture.detectChanges();
    let modelRetryButton = fixture.debugElement.query(By.css('#model_settings_retry'));
    expect(modelRetryButton).toBeTruthy();
    modelRetryButton.triggerEventHandler('click');
    expect(modelBackendService.retryAll).toHaveBeenCalled();
  });

  it('should unsubscribe from all observables', () => {
    fixture.destroy();
    expect(model_settings_observable.observed).toBeFalse();
    expect(model_settings_progress_observable.observed).toBeFalse();
  });
});
