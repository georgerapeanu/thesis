import { Component, OnDestroy, OnInit } from '@angular/core';
import { ModelBackendService } from '../../services/model-backend.service';
import { CommonModule } from '@angular/common';
import { MatButtonModule} from '@angular/material/button';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { Subscription } from 'rxjs';
import { MatIconModule } from '@angular/material/icon';
import { TopKDTO } from '../../dto/topkDTO';
import { ProgressEnum } from '../../enums/ProgressEnum';


@Component({
  selector: 'app-see-topk',
  standalone: true,
  imports: [CommonModule, MatButtonModule, MatProgressSpinnerModule, MatIconModule],
  templateUrl: './see-topk.component.html',
  styleUrl: './see-topk.component.css'
})
export class SeeTopkComponent implements OnInit, OnDestroy {
  public topk: Array<[number, string]> = [];

  topkSubscription: Subscription | null = null;
  state = ProgressEnum.LOADING;

  constructor(private modelBackendService: ModelBackendService) {
    this.modelBackendService = modelBackendService;
  }

  ngOnDestroy(): void {
    this.topkSubscription?.unsubscribe();
  }

  ngOnInit(): void {
    this.topkSubscription = this.modelBackendService.getTopKObservable().subscribe({
      next: (topk_dto: TopKDTO) => {
        this.topk = topk_dto.topk.map((value) => [Math.round(value[0] * 10000) / 100, value[1]]);
        this.state = topk_dto.state;
      },
    });
  }

  isDisabled(prob_token: [number, string]): boolean {
    return (prob_token[1] === '</s>');
  }

  onClick(prob_token: [number, string]) {
    if(this.isDisabled(prob_token)) {
      return ;
    }
    let [_, token] = prob_token;
    let prefix = this.modelBackendService.get_prefix();
    token = token.replaceAll('▁', ' ');
    if(prefix.length === 0) {
      token = token.trimStart();
    }
    prefix += token;
    this.modelBackendService.set_prefix(prefix);
    this.topk = [];
  }

  onRetryTopK() {
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
