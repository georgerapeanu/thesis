import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ModelSettingsComponent } from './model-settings.component';

describe('ModelSettingsComponent', () => {
  let component: ModelSettingsComponent;
  let fixture: ComponentFixture<ModelSettingsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ModelSettingsComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(ModelSettingsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
