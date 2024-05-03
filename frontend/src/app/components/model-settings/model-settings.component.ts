import { Component, OnDestroy, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatSliderModule } from '@angular/material/slider';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatSlideToggleModule } from '@angular/material/slide-toggle';
import { MatRadioModule } from '@angular/material/radio';
import { CommonModule } from '@angular/common';
import { ModelBackendService } from '../../services/model-backend.service';
import { Subscription } from 'rxjs';
import { ModelSettingsDTO } from '../../dto/modelSettingsDTO';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatButtonModule } from '@angular/material/button';
import { ProgressEnum } from '../../enums/ProgressEnum';

@Component({
  selector: 'app-model-settings',
  standalone: true,
  imports: [CommonModule, MatSliderModule, MatProgressSpinnerModule, MatButtonModule, MatIconModule, FormsModule, MatTooltipModule, MatSlideToggleModule, MatRadioModule],
  templateUrl: './model-settings.component.html',
  styleUrl: './model-settings.component.css'
})
export class ModelSettingsComponent implements OnInit, OnDestroy {
  modelSettings: ModelSettingsDTO | null = null;
  modelSettingsSubscription: Subscription | null = null;
  modelSettingsLoadedSubscription: Subscription | null = null;
  state: ProgressEnum = ProgressEnum.LOADING;

  constructor(private modelBackendService: ModelBackendService) {
    this.modelBackendService = modelBackendService;
  }

  ngOnDestroy(): void {
    this.modelSettingsSubscription?.unsubscribe();
    this.modelSettingsLoadedSubscription?.unsubscribe();
  }

  ngOnInit(): void {
    this.modelSettingsSubscription = this.modelBackendService.getModelSettingsDistinctUntilChangedObservable().subscribe((settings: ModelSettingsDTO) => {
      this.modelSettings = settings.clone();
    });
    this.modelSettingsLoadedSubscription = this.modelBackendService.getModelSettingsProgressObservable().subscribe((state: ProgressEnum) => {
      this.state = state;
    });
  }

  formatLabel(value: number): string {
    return `${value}`;
  }

  updateCommentaryType() {
    if(this.modelSettings === null) {
      return;
    }
    this.modelBackendService.set_commentary_type(this.modelSettings.target_type);

  }

  updateSample() {
    if(this.modelSettings === null) {
      return;
    }
    this.modelBackendService.set_doSample(this.modelSettings.do_sample);
  }

  updateTemperature() {
    if(this.modelSettings === null) {
      return;
    }
    this.modelBackendService.set_temperature(this.modelSettings.temperature);
  }

  updateMaxNewTokens() {
    if(this.modelSettings === null) {
      return;
    }
    this.modelBackendService.set_max_new_tokens(this.modelSettings.max_new_tokens);
  }

  updatePrefix() {
    if(this.modelSettings === null) {
      return;
    }
    this.modelBackendService.set_prefix(this.modelSettings.prefix);
  }

  onRetryModelSettings() {
    this.modelBackendService.retryAll();
  }

  isLoading(): boolean {
    return this.state === ProgressEnum.LOADING;
  }

  isLoaded(): boolean {
    return this.state === ProgressEnum.LOADED;
  }

  isFailed(): boolean {
    return this.state === ProgressEnum.FAILED;
  }
}
