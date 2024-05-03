enum TopKStateEnum {
  LOADED,
  LOADING,
  FAILED
}

export class TopKDTO {
  static State = TopKStateEnum;

  topk: Array<[number, string]>;
  state: TopKStateEnum;


  constructor(
    topk: Array<[number, string]>,
    state: TopKStateEnum
  ) {
    this.topk = topk;
    this.state = state;
  }
}
