import { Component, Input, OnInit } from '@angular/core';
import { GameStateService } from '../../services/game-state.service';
import { CommonModule } from '@angular/common';
import { Chess, Square, Move, KING } from 'chess.js';

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

  constructor(
    private boardStateService: GameStateService
  ) {
    this.boardStateService = boardStateService;
    this.boardStateService.get_observable_game().subscribe((game: Chess): void => {
      this.updateComponentState(game, this.flipped, this.focusedSquare);
    });
  }

  ngOnInit() {
    this.boardStateService.set_current_fen("1nbqkbnr/rpp2pPp/8/3pP3/8/p4NPB/PPP1P2P/RNBQK2R w KQk d6 0 11");
  }

  private updateComponentState(
    game: Chess | null,
    flipped: boolean,
    focusedSquare: string | null
  ): void {
    this.lastGame = game;
    this.flipped = flipped;
    this.focusedSquare = focusedSquare;
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
    this.updateComponentState(this.lastGame, this.flipped, square);
  }

  public unfocusCurrentSquare(): void {
    this.updateComponentState(this.lastGame, this.flipped, null);
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
          this.unfocusCurrentSquare();
          this.boardStateService.move(move_move);
          this.lastMove = move_move;
          return;
        }
      }
      //TODO promotions
    } else if(square !== this.focusedSquare && this.lastGame?.get(square as Square) && this.lastGame?.get(square as Square).color === this.lastGame?.turn()) {
      this.focusSquare(square);
    } else {
      this.unfocusCurrentSquare();
    }
  }
  public isPartOfLastMove(square: string): boolean {
    return [this.lastMove?.from, this.lastMove?.to].includes(square as Square);
  }

  onDragOver(e: DragEvent) {
    e.preventDefault();
  }
  //tricky moves: castling, promotions
  // TODO implement check too
}
