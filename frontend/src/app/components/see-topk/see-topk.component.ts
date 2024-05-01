import { Component, OnDestroy, OnInit } from '@angular/core';
import { ModelBackendService } from '../../services/model-backend.service';
import { CommonModule } from '@angular/common';
import {MatButtonModule} from '@angular/material/button';
import {MatProgressSpinnerModule} from '@angular/material/progress-spinner';
import { Subscription } from 'rxjs';

@Component({
  selector: 'app-see-topk',
  standalone: true,
  imports: [CommonModule, MatButtonModule, MatProgressSpinnerModule],
  templateUrl: './see-topk.component.html',
  styleUrl: './see-topk.component.css'
})
export class SeeTopkComponent implements OnInit, OnDestroy {

  public topk: Array<[number, string]> = [];
  public loading = false;

  topkSubscription: Subscription | null = null;
  loadingSubscription: Subscription | null = null;

  constructor(private modelBackendService: ModelBackendService) {
    this.modelBackendService = modelBackendService;
  }

  ngOnDestroy(): void {
    this.topkSubscription?.unsubscribe();
    this.loadingSubscription?.unsubscribe();
  }

  ngOnInit(): void {
    this.topkSubscription = this.modelBackendService.getTopKLoadingObservable().subscribe((loading) => {
      this.loading = loading;
      if(loading === true) {
        this.topk = [];
      }
    });

    this.loadingSubscription = this.modelBackendService.getTopKObservable().subscribe((topk) => {
      this.loading = false;
      this.topk = topk.map((value) => [Math.round(value[0] * 10000) / 100, value[1]]);
    })
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
}
