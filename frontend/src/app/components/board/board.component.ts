import { Component, Input, OnInit } from '@angular/core';
import { GameStateService } from '../../services/game-state.service';
import { CommonModule } from '@angular/common';
import { Chess, Square, Move, KING, BLACK, PieceSymbol } from 'chess.js';

@Component({
  selector: 'app-board',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './board.component.html',
  styleUrl: './board.component.css'
})
export class BoardComponent implements OnInit {
  @Input() flipped: boolean = false;
  ranks: Array<string> = [];
  files: Array<string> = [];
  squares: Array<string> = [];
  lastGame: Chess | null = null;

  focusedSquare: string | null = null;
  shownMoves: Array<string> = [];
  lastMove: Move | null = null;
  pendingPromotionMove: Move | null = null;

  constructor(
    private gameStateService: GameStateService
  ) {
    this.gameStateService = gameStateService;
  }

  ngOnInit() {
    this.gameStateService.get_observable_state().subscribe((_game_index: [Chess, number]): void => {
      let actual_game = this.gameStateService.get_chess_game_at_index();
      this.updateComponentState(actual_game, this.flipped, this.focusedSquare, this.pendingPromotionMove);
    });
  }

  private updateComponentState(
    game: Chess | null,
    flipped: boolean,
    focusedSquare: string | null,
    pendingPromotionMove: Move | null
  ): void {
    this.lastGame = game;
    this.flipped = flipped;
    this.focusedSquare = focusedSquare;
    this.pendingPromotionMove = pendingPromotionMove;
    this.lastMove = game?.history({verbose: true}).slice(-1)[0] || null;
    this.shownMoves = [];

    this.ranks = Array(8).fill(1).map((_x, i) => (i + 1).toString()).reverse();
    this.files = Array(8).fill(1).map((_x, i) => String.fromCharCode(97 + i));
    if(this.flipped) {
      this.files.reverse();
      this.ranks.reverse();
    }
    this.squares = [];
    for(let i = 0; i < this.ranks.length; i++) {
      for(let j = 0; j < this.files.length; j++) {
        this.squares.push(this.files[j] + this.ranks[i]);
      }
    }
    if(this.lastGame && this.focusedSquare) {
      for(const move of this.lastGame?.moves({verbose: true, square: (this.focusedSquare as Square)})) {
        this.shownMoves.push((move as any as Move).to);
      }
    }
  }

  public getImageForCell(square: string): string | null {
    var piece = this.lastGame?.get(square as Square);
    if(!piece) {
      return null;
    }

    return piece.color + piece.type.toString().toUpperCase();
  }

  public isSquareByIndexBlack(index: number): boolean {
    return !(index % 2 === Math.floor((index / 8)) % 2);
  }

  public focusSquare(square: string): void {
    this.updateComponentState(this.lastGame, this.flipped, square, null);
  }

  public unfocusCurrentSquare(): void {
    this.updateComponentState(this.lastGame, this.flipped, null, this.pendingPromotionMove);
  }

  public isLegalMove(square: string): boolean {
    return this.shownMoves.includes(square);
  }

  public isCapture(square: string): boolean {
    return this.isLegalMove(square) && !!this.lastGame?.get(square as Square);
  }

  public isCheck(square: string): boolean {
    var piece = this.lastGame?.get(square as Square);
    if(!piece) {
      return false;
    }

    return (piece.type === KING &&  piece.color === this.lastGame?.turn() && (this.lastGame?.inCheck() || false));
  }

  public clickSquare(square: string): void {
    if(this.shownMoves.includes(square)) {
      for(const move of this.lastGame?.moves({verbose: true, square: (this.focusedSquare as Square)})!) {
        let move_move = (move as any as Move);
        if(move_move.to === square) {
          if('promotion' in move_move) {
            this.pendingPromotionMove = move_move;
          } else {
            this.unfocusCurrentSquare();
            this.lastMove = move_move;
            this.gameStateService.move(move_move);
          }
          return;
        }
      }
    } else if(square !== this.focusedSquare && this.lastGame?.get(square as Square) && this.lastGame?.get(square as Square).color === this.lastGame?.turn()) {
      this.focusSquare(square);
    } else {
      this.unfocusCurrentSquare();
    }
  }

  public isPartOfLastMove(square: string): boolean {
    return [this.lastMove?.from, this.lastMove?.to].includes(square as Square);
  }

  public cancelPromotion() {
    this.unfocusCurrentSquare();
    this.pendingPromotionMove = null;
  }

  onDragOver(e: DragEvent) {
    e.preventDefault();
  }

  public getPromotionPieces(): Array<string> {
    if(!this.pendingPromotionMove) {
      throw "Unexpected promotion";
    }
    var answer = ['Q', 'N', 'R', 'B'].map(x => this.pendingPromotionMove!.color + x);
    if(this.flipped !== (this.pendingPromotionMove.color === BLACK)) {
      answer.reverse();
    }
    return answer;
  }

  public promote(piece: string): void {
    if(!this.pendingPromotionMove) {
      throw "Unexpected promotion";
    }
    let promotion = piece[1].toLowerCase() as PieceSymbol;
    for(const move of this.lastGame?.moves({verbose: true, square: (this.focusedSquare as Square)})!) {
      if(move.to === this.pendingPromotionMove?.to && move.from === this.pendingPromotionMove?.from && move.promotion === promotion) {
        this.lastMove = move;
        this.cancelPromotion();
        this.gameStateService.move(move);
      }
    }
  }

  public undo() {
    console.log("aaaaaa");
    this.gameStateService.undo();
  }

  public redo() {
    this.gameStateService.redo();
  }

  public top() {
    this.gameStateService.seek(0);
  }

  public bottom() {
    this.gameStateService.seek(this.gameStateService.get_current_state()[0].history().length);
  }

}
