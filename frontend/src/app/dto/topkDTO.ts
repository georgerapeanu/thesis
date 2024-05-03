import { ProgressEnum } from "../enums/ProgressEnum";

export class TopKDTO {

  topk: Array<[number, string]>;
  state: ProgressEnum;


  constructor(
    topk: Array<[number, string]>,
    state: ProgressEnum
  ) {
    this.topk = topk;
    this.state = state;
  }
}
