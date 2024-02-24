# train a sentencepiece model on it
# the settings here are (best effort) those used for training Llama 2
import os
import sentencepiece as spm

VOCAB_SIZE = 1500

if __name__ == '__main__':
    options = dict(
      # input spec
      input="../artifacts/commentaries.txt",
      input_format="text",
      # output spec
      model_prefix=f"../artifacts/sp{VOCAB_SIZE}", # output filename prefix
      # algorithm spec
      # BPE alg
      model_type="bpe",
      vocab_size=VOCAB_SIZE,
      # normalization
      normalization_rule_name="identity", # ew, turn off normalization
      remove_extra_whitespaces=False,
      input_sentence_size=200000000, # max number of training sentences
      max_sentence_length=4192, # max number of bytes per sentence
      seed_sentencepiece_size=1000000,
      shuffle_input_sentence=True,
      # rare word treatment
      character_coverage=0.99995,
      byte_fallback=True,
      # merge rules
      split_digits=True,
      split_by_unicode_script=True,
      split_by_whitespace=True,
      split_by_number=True,
      max_sentencepiece_length=16,
      add_dummy_prefix=True,
      allow_whitespace_only_pieces=True,
      # special tokens
      unk_id=0, # the UNK token MUST exist
      bos_id=1, # the others are optional, set to -1 to turn off
      eos_id=2,
      pad_id=3,
      # systems
      num_threads=os.cpu_count(), # use ~all system resources
    )

    spm.SentencePieceTrainer.train(**options)
