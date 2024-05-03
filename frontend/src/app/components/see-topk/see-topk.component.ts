import { Component, OnDestroy, OnInit } from '@angular/core';
import { ModelBackendService } from '../../services/model-backend.service';
import { CommonModule } from '@angular/common';
import { MatButtonModule} from '@angular/material/button';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { Subscription } from 'rxjs';
import { MatIconModule } from '@angular/material/icon';
import { TopKDTO } from '../../dto/topkDTO';


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
  state = TopKDTO.State.LOADING;

  constructor(private modelBackendService: ModelBackendService) {
    this.modelBackendService = modelBackendService;
  }

  ngOnDestroy(): void {
    this.topkSubscription?.unsubscribe();
  }

  ngOnInit(): void {
    this.topkSubscription = this.modelBackendService.getTopKObservable().subscribe({
      next: (topk_dto: TopKDTO) => {
        this.state = topk_dto.state;
        this.topk = topk_dto.topk.map((value) => [Math.round(value[0] * 10000) / 100, value[1]]);
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
    let prefix = this.modelBackendService.prefix;
    token = token.replaceAll('▁', ' ');
    if(prefix.length === 0) {
      token = token.trimStart();
    }
    prefix += token;
    this.modelBackendService.prefix = prefix;
    this.topk = [];
  }

  onRetryTopK() {
    this.modelBackendService.manualRetryTopK();
  }

  isLoading(): boolean {
    return this.state === TopKDTO.State.LOADING;
  }

  isLoaded(): boolean {
    return this.state === TopKDTO.State.LOADED;
  }

  isFailed(): boolean {
    return this.state === TopKDTO.State.FAILED;
  }
}
