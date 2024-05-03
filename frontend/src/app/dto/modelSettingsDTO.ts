
export class ModelSettingsDTO {
  temperature: number;
  do_sample: boolean;
  target_type: string;
  max_new_tokens: number;
  prefix: string;
  commentary_types: Array<[string, string]>;
  min_temperature: number;
  max_temperature: number;
  max_max_new_tokens: number;

  constructor(constructor_dict: {
    temperature: number,
    do_sample: boolean,
    target_type: string,
    max_new_tokens: number,
    prefix: string,
    commentary_types: Array<[string, string]>,
    min_temperature: number,
    max_temperature: number,
    max_max_new_tokens: number
  } ) {
    this.temperature = constructor_dict.temperature;
    this.do_sample = constructor_dict.do_sample;
    this.target_type = constructor_dict.target_type;
    this.max_new_tokens = constructor_dict.max_new_tokens;
    this.prefix = constructor_dict.prefix;
    this.commentary_types = constructor_dict.commentary_types;
    this.min_temperature = constructor_dict.min_temperature;
    this.max_temperature = constructor_dict.max_temperature;
    this.max_max_new_tokens = constructor_dict.max_max_new_tokens;

  }

  clone(): ModelSettingsDTO {
    return new ModelSettingsDTO({
      temperature: this.temperature,
      do_sample: this.do_sample,
      target_type: this.target_type,
      max_new_tokens: this.max_new_tokens,
      prefix: this.prefix,
      commentary_types: JSON.parse(JSON.stringify(this.commentary_types)),
      min_temperature: this.min_temperature,
      max_temperature: this.max_temperature,
      max_max_new_tokens: this.max_max_new_tokens
    });
  }
}
