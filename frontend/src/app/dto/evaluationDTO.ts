export class EvaluationDTO {
  public isMate: boolean;
  public score: number;
  public depth: number;


  constructor(
    isMate: boolean,
    score: number,
    depth: number
  ) {
    this.isMate = isMate;
    this.score = score;
    this.depth = depth;
  }
}
