import { Component, OnDestroy, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import {MatSliderModule} from '@angular/material/slider';
import {MatTooltipModule} from '@angular/material/tooltip';
import {MatSlideToggleModule} from '@angular/material/slide-toggle';
import {MatRadioModule} from '@angular/material/radio';
import { CommonModule } from '@angular/common';
import { ModelBackendService } from '../../services/model-backend.service';
import { Subscription } from 'rxjs';
import { ModelSettingsDTO } from '../../dto/modelSettingsDTO';

@Component({
  selector: 'app-model-settings',
  standalone: true,
  imports: [CommonModule, MatSliderModule, FormsModule, MatTooltipModule, MatSlideToggleModule, MatRadioModule],
  templateUrl: './model-settings.component.html',
  styleUrl: './model-settings.component.css'
})
export class ModelSettingsComponent implements OnInit, OnDestroy {
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
  modelSettingsSubscription: Subscription | null = null;

  constructor(private modelBackendService: ModelBackendService) {
    this.modelBackendService = modelBackendService;
  }

  ngOnDestroy(): void {
    this.modelSettingsSubscription?.unsubscribe();
  }

  ngOnInit(): void {
    this.temperature = this.modelBackendService.temperature;
    this.sample = this.modelBackendService.doSample;
    this.commentary_type = this.modelBackendService.commentary_type;
    this.max_new_tokens = this.modelBackendService.max_new_tokens;
    this.prefix = this.modelBackendService.prefix;

    this.modelSettingsSubscription = this.modelBackendService.getModelSettingsDistinctUntilChangedObservable().subscribe((settings: ModelSettingsDTO) => {
      this.temperature = settings.temperature;
      this.sample = settings.do_sample;
      this.commentary_type = settings.target_type;
      this.max_new_tokens = settings.max_new_tokens;
      this.prefix = settings.prefix;
    });
  }

  formatLabel(value: number): string {
    return `${value}`;
  }

  updateCommentaryType() {
    this.modelBackendService.commentary_type = this.commentary_type;
  }

  updateSample() {
    this.modelBackendService.doSample = this.sample;
  }

  updateTemperature() {
    this.modelBackendService.temperature = this.temperature;
  }

  updateMaxNewTokens() {
    this.modelBackendService.max_new_tokens = this.max_new_tokens;
  }

  updatePrefix() {
    this.modelBackendService.prefix = this.prefix;
  }
}
