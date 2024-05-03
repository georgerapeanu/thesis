
export class ModelSettingsDTO {
  temperature: number;
  do_sample: boolean;
  target_type: string;
  max_new_tokens: number;
  prefix: string;

  constructor(constructor_dict: {
    temperature: number,
    do_sample: boolean,
    target_type: string,
    max_new_tokens: number,
    prefix: string,
  } ) {
    this.temperature = constructor_dict.temperature;
    this.do_sample = constructor_dict.do_sample;
    this.target_type = constructor_dict.target_type;
    this.max_new_tokens = constructor_dict.max_new_tokens;
    this.prefix = constructor_dict.prefix;

  }

  clone(): ModelSettingsDTO {
    return new ModelSettingsDTO({
      temperature: this.temperature,
      do_sample: this.do_sample,
      target_type: this.target_type,
      max_new_tokens: this.max_new_tokens,
      prefix: this.prefix
    });
  }
}
