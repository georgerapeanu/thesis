import { TestBed } from '@angular/core/testing';

import { ModelBackendService } from './model-backend.service';
import { HttpClientModule } from '@angular/common/http';

describe('ModelBackendService', () => {
  let service: ModelBackendService;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [HttpClientModule]
    });
    service = TestBed.inject(ModelBackendService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
