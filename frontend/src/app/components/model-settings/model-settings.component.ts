import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import {MatSliderModule} from '@angular/material/slider';
import {MatTooltipModule} from '@angular/material/tooltip';
import {MatSlideToggleModule} from '@angular/material/slide-toggle';
import {MatRadioModule} from '@angular/material/radio';
import { CommonModule } from '@angular/common';
import { ModelBackendService } from '../../services/model-backend.service';

@Component({
  selector: 'app-model-settings',
  standalone: true,
  imports: [CommonModule, MatSliderModule, FormsModule, MatTooltipModule, MatSlideToggleModule, MatRadioModule],
  templateUrl: './model-settings.component.html',
  styleUrl: './model-settings.component.css'
})
export class ModelSettingsComponent {
  public temperature_min = 0.1;
  public temperature_max = 3;
  public temperature = 1.0;
  public sample = false;
  public commentary_type = "";
  public max_new_tokens_min = 100;
  public max_new_tokens_max= 1000;
  public max_new_tokens_step = 5;
  public max_new_tokens = 500;
  public prefix = "";

  constructor(private modelBackendService: ModelBackendService) {
    this.modelBackendService = modelBackendService;
  }

  formatLabel(value: number): string {
    return `${value}`;
  }

  updateSettings() {
    this.modelBackendService.commentary_type = this.commentary_type;
    this.modelBackendService.doSample = this.sample;
    this.modelBackendService.temperature = this.temperature;
    this.modelBackendService.max_new_tokens = this.max_new_tokens;
    this.modelBackendService.prefix = this.prefix;
  }
}
