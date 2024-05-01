import { Component, OnInit } from '@angular/core';
import { ModelBackendService } from '../../services/model-backend.service';
import { CommonModule } from '@angular/common';
import {MatButtonModule} from '@angular/material/button';

@Component({
  selector: 'app-see-topk',
  standalone: true,
  imports: [CommonModule, MatButtonModule],
  templateUrl: './see-topk.component.html',
  styleUrl: './see-topk.component.css'
})
export class SeeTopkComponent implements OnInit {

  public topk: Array<[number, string]> = [];

  constructor(private modelBackendService: ModelBackendService) {
    this.modelBackendService = modelBackendService;
  }

  ngOnInit(): void {
    this.modelBackendService.getTopKObservable().subscribe((topk) => {
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
    token = token.replace('▁', ' ');
    if(prefix.length === 0) {
      token = token.trimStart();
    }
    prefix += token;
    this.modelBackendService.prefix = prefix;
  }
}
