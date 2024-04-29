import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SeeTopkComponent } from './see-topk.component';

describe('SeeTopkComponent', () => {
  let component: SeeTopkComponent;
  let fixture: ComponentFixture<SeeTopkComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SeeTopkComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(SeeTopkComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
