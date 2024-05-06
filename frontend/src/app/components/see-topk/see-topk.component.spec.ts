import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SeeTopkComponent } from './see-topk.component';
import { ModelBackendService } from '../../services/model-backend.service';
import { Subject } from 'rxjs';
import { TopKDTO } from '../../dto/topkDTO';
import { ProgressEnum } from '../../enums/ProgressEnum';
import { By } from '@angular/platform-browser';

describe('SeeTopkComponent', () => {
  let component: SeeTopkComponent;
  let fixture: ComponentFixture<SeeTopkComponent>;
  let modelBackendService: jasmine.SpyObj<ModelBackendService>;
  let topKObservable: Subject<TopKDTO>;

  beforeEach(async () => {
    modelBackendService = jasmine.createSpyObj('ModelBackendService', [
      'getTopKObservable',
      'retryAll',
      'get_prefix',
      'set_prefix'
    ]);

    topKObservable = new Subject<TopKDTO>();
    modelBackendService.getTopKObservable.and.returnValue(topKObservable);

    await TestBed.configureTestingModule({
      imports: [SeeTopkComponent],
      providers: [
        { provide: ModelBackendService, useValue: modelBackendService},
      ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(SeeTopkComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should update topk according to observable', () => {
    topKObservable.next(new TopKDTO([[1, "a"]], ProgressEnum.LOADED));
    //100 due to transforming probabilities to percentages
    expect(component.topk).toEqual([[100, "a"]]);
    expect(component.state).toEqual(ProgressEnum.LOADED);
  });

  it('should return true for disabled only if token is eos', () => {
    expect(component.isDisabled([0.3, '</s>'])).toBeTrue();
    expect(component.isDisabled([0.3, 'a'])).toBeFalse();
  });

  it('should call retryAll in service when requested', () => {
    component.onRetryTopK();
    expect(modelBackendService.retryAll).toHaveBeenCalled();
  });

  it('should call retryAll in service when requested DOM', () => {
    topKObservable.next(new TopKDTO([], ProgressEnum.FAILED));
    fixture.detectChanges();

    let retryButton = fixture.debugElement.query(By.css('#topk_retry_button'));

    expect(retryButton).toBeTruthy();
    retryButton.triggerEventHandler('click');

    expect(modelBackendService.retryAll).toHaveBeenCalled();
  });

  it('should not modify prefix onclick for eos', () => {
    modelBackendService.get_prefix.and.returnValue("test");
    component.onClick([1, "</s>"]);
    expect(modelBackendService.set_prefix).not.toHaveBeenCalled();
  });

  it('should append the clicked token to prefix', () => {
    modelBackendService.get_prefix.and.returnValue("test");
    component.onClick([1, "a"]);
    expect(modelBackendService.set_prefix).toHaveBeenCalledWith("testa");
  });

  it('should trim token start on empty prefix', () => {
    modelBackendService.get_prefix.and.returnValue("");
    component.onClick([1, " a"]);
    expect(modelBackendService.set_prefix).toHaveBeenCalledWith("a");
  });

  it('should unsubscribe from all observables on destroy', () => {
    fixture.destroy();
    expect(topKObservable.observed).toBeFalse();
  });
});
