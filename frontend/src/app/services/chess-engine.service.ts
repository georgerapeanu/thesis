import { Injectable } from '@angular/core';
import { BLACK, Chess } from 'chess.js';
import { BehaviorSubject, Observable, filter, map, merge, of, skipWhile, take, takeWhile } from 'rxjs';
import { EvaluationDTO } from '../dto/evaluationDTO';

@Injectable({
  providedIn: 'root'
})
export class ChessEngineService {
  stockfish: Worker | null = null;
  request_number = 0;
  messageBehaviorSubject = new BehaviorSubject<[number, string]>([this.request_number, "canStart"]);
  messageToEvaluationRegex = new RegExp(" depth ([0-9]*).*score ([^ ]*) ([^ ]*) nodes");
  engine_busy = false;

  constructor() {
    var wasmSupported = typeof WebAssembly === 'object' && WebAssembly.validate(Uint8Array.of(0x0, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00));

    if (typeof Worker !== 'undefined') {
      this.stockfish = new Worker(wasmSupported ? 'assets/js/stockfish.wasm.js' : 'assets/js/stockfish.js');
      this.stockfish.addEventListener('message', (e) => {
        let data: [number, string] = [this.request_number, e.data];
        if(data[1].startsWith("bestmove")) {
          this.engine_busy = false;
          data = [this.request_number, "canStart"];
        }
        this.messageBehaviorSubject.next(data);
      });
    }

  }



  requestEvaluation(chess: Chess): Observable<EvaluationDTO | null> {
    if(this.stockfish === null) {
      return of(null);
    }
    this.request_number += 1;
    let own_request_number = this.request_number;
    let turn = chess.turn();
    let fen = chess.fen();

    let sourceMessageObservable = this.messageBehaviorSubject.asObservable();
    if(!this.engine_busy) {
      sourceMessageObservable = merge(sourceMessageObservable, of<[number, string]>([own_request_number, "canStart"]));
    }

    this.stockfish.postMessage('stop');

    let my_messages_observable = sourceMessageObservable
    .pipe(skipWhile((request_message) => request_message[0] < own_request_number))
    .pipe(takeWhile((request_message) => request_message[0] === own_request_number))
    .pipe(map((request_message) => request_message[1].trim()))

    my_messages_observable
    .pipe(filter((message) => message.startsWith('canStart')))
    .pipe(take(1))
    .subscribe({
      next: (_value) => {
        this.engine_busy = true;
        this.stockfish?.postMessage(`position fen ${fen}`);
        this.stockfish?.postMessage('go depth 20');
      }
    });

    let my_evaluations_observable = my_messages_observable
    .pipe(filter((message) => message.startsWith('info')))
    .pipe(map((message) => {
      let groups = message.match(this.messageToEvaluationRegex);
      let depth = parseInt(groups![1]);
      let score_type = groups![2];
      let score = parseInt(groups![3]);

      if(turn === BLACK) {
        score *= -1;
      }
      if(score_type === "cp") {
        score = score / 100;
      }

      return new EvaluationDTO((score_type !== "cp"), score, depth);
    }));

    return my_evaluations_observable;
  }
}
