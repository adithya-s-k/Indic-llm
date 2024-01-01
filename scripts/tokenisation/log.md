
kannada_sentence_corpus_wikipedia-31.4k.txt
`python3 ./scripts/tokenisation/train.py --input-file "./temp-corpus/kannada_sentence_corpus_wikipedia-31.4k.txt`
```
sentencepiece_trainer.cc(77) LOG(INFO) Starts training with : 
trainer_spec {
  input: ./temp-corpus/kannada_sentence_corpus_wikipedia-31.4k.txt
  input_format: 
  model_prefix: kannada_sp
  model_type: UNIGRAM
  vocab_size: 20000
  self_test_sample_size: 0
  character_coverage: 1
  input_sentence_size: 0
  shuffle_input_sentence: 1
  seed_sentencepiece_size: 1000000
  shrinking_factor: 0.75
  max_sentence_length: 4192
  num_threads: 16
  num_sub_iterations: 2
  max_sentencepiece_length: 16
  split_by_unicode_script: 1
  split_by_number: 1
  split_by_whitespace: 1
  split_digits: 0
  pretokenization_delimiter: 
  treat_whitespace_as_suffix: 0
  allow_whitespace_only_pieces: 0
  required_chars: 
  byte_fallback: 0
  vocabulary_output_piece_score: 1
  train_extremely_large_corpus: 1
  hard_vocab_limit: 1
  use_all_vocab: 0
  unk_id: 0
  bos_id: 1
  eos_id: 2
  pad_id: -1
  unk_piece: <unk>
  bos_piece: <s>
  eos_piece: </s>
  pad_piece: <pad>
  unk_surface:  ⁇ 
  enable_differential_privacy: 0
  differential_privacy_noise_level: 0
  differential_privacy_clipping_threshold: 0
}
normalizer_spec {
  name: nmt_nfkc
  add_dummy_prefix: 1
  remove_extra_whitespaces: 1
  escape_whitespaces: 1
  normalization_rule_tsv: 
}
denormalizer_spec {}
trainer_interface.cc(351) LOG(INFO) SentenceIterator is not specified. Using MultiFileSentenceIterator.
trainer_interface.cc(183) LOG(INFO) Loading corpus: ./temp-corpus/kannada_sentence_corpus_wikipedia-31.4k.txt
trainer_interface.cc(378) LOG(WARNING) Found too long line (4389 > 4192).
trainer_interface.cc(380) LOG(WARNING) Too long lines are skipped in the training.
trainer_interface.cc(381) LOG(WARNING) The maximum length can be changed with --max_sentence_length=<size> flag.
trainer_interface.cc(407) LOG(INFO) Loaded all 913102 sentences
trainer_interface.cc(414) LOG(INFO) Skipped 3978 too long sentences.
trainer_interface.cc(423) LOG(INFO) Adding meta_piece: <unk>
trainer_interface.cc(423) LOG(INFO) Adding meta_piece: <s>
trainer_interface.cc(423) LOG(INFO) Adding meta_piece: </s>
trainer_interface.cc(428) LOG(INFO) Normalizing sentences...
trainer_interface.cc(537) LOG(INFO) all chars count=139563157
trainer_interface.cc(548) LOG(INFO) Done: 100% characters are covered.
trainer_interface.cc(558) LOG(INFO) Alphabet size=3048
trainer_interface.cc(559) LOG(INFO) Final character coverage=1
trainer_interface.cc(591) LOG(INFO) Done! preprocessed 901799 sentences.
unigram_model_trainer.cc(222) LOG(INFO) Making suffix array...
unigram_model_trainer.cc(226) LOG(INFO) Extracting frequent sub strings... node_num=69715645
unigram_model_trainer.cc(274) LOG(INFO) Initialized 1003048 seed sentencepieces
trainer_interface.cc(597) LOG(INFO) Tokenizing input sentences with whitespace: 901799
trainer_interface.cc(608) LOG(INFO) Done! 2117484
unigram_model_trainer.cc(564) LOG(INFO) Using 2117484 sentences for EM training
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=0 size=474378 obj=16.2028 num_tokens=5533272 num_tokens/piece=11.6643
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=1 size=388251 obj=13.374 num_tokens=5547434 num_tokens/piece=14.2883
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=0 size=290956 obj=13.3364 num_tokens=5614671 num_tokens/piece=19.2973
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=1 size=289718 obj=13.3176 num_tokens=5635913 num_tokens/piece=19.4531
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=0 size=217281 obj=13.3564 num_tokens=5750812 num_tokens/piece=26.4672
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=1 size=217259 obj=13.3409 num_tokens=5751347 num_tokens/piece=26.4723
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=0 size=162944 obj=13.4343 num_tokens=5944920 num_tokens/piece=36.4844
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=1 size=162941 obj=13.4054 num_tokens=5945209 num_tokens/piece=36.4869
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=0 size=122205 obj=13.5665 num_tokens=6168303 num_tokens/piece=50.475
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=1 size=122205 obj=13.5226 num_tokens=6168904 num_tokens/piece=50.48
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=0 size=91653 obj=13.7347 num_tokens=6404078 num_tokens/piece=69.8731
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=1 size=91653 obj=13.6821 num_tokens=6404495 num_tokens/piece=69.8776
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=0 size=68739 obj=13.938 num_tokens=6656318 num_tokens/piece=96.8347
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=1 size=68739 obj=13.8786 num_tokens=6656956 num_tokens/piece=96.8439
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=0 size=51554 obj=14.1779 num_tokens=6925656 num_tokens/piece=134.338
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=1 size=51554 obj=14.1119 num_tokens=6926069 num_tokens/piece=134.346
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=0 size=38665 obj=14.4625 num_tokens=7213685 num_tokens/piece=186.569
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=1 size=38665 obj=14.3872 num_tokens=7214190 num_tokens/piece=186.582
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=0 size=28998 obj=14.7975 num_tokens=7536253 num_tokens/piece=259.889
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=1 size=28998 obj=14.7087 num_tokens=7536690 num_tokens/piece=259.904
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=0 size=22000 obj=15.1659 num_tokens=7886763 num_tokens/piece=358.489
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=1 size=22000 obj=15.0656 num_tokens=7887067 num_tokens/piece=358.503
trainer_interface.cc(686) LOG(INFO) Saving model: kannada_sp.model
trainer_interface.cc(698) LOG(INFO) Saving vocabs: kannada_sp.vocab
Total time taken to train the model: 1.99 minutes
```

kannada_sentence_corpus_CulturaX_1GB.txt
`python3 ./scripts/tokenisation/train.py --input-file "./corpus/kannada_sentence_corpus_CulturaX_1GB.txt"`
```
sentencepiece_trainer.cc(77) LOG(INFO) Starts training with : 
trainer_spec {
  input: ./corpus/kannada_sentence_corpus_CulturaX_1GB.txt
  input_format: 
  model_prefix: kannada_sp
  model_type: UNIGRAM
  vocab_size: 20000
  self_test_sample_size: 0
  character_coverage: 1
  input_sentence_size: 0
  shuffle_input_sentence: 1
  seed_sentencepiece_size: 1000000
  shrinking_factor: 0.75
  max_sentence_length: 4192
  num_threads: 16
  num_sub_iterations: 2
  max_sentencepiece_length: 16
  split_by_unicode_script: 1
  split_by_number: 1
  split_by_whitespace: 1
  split_digits: 0
  pretokenization_delimiter: 
  treat_whitespace_as_suffix: 0
  allow_whitespace_only_pieces: 0
  required_chars: 
  byte_fallback: 0
  vocabulary_output_piece_score: 1
  train_extremely_large_corpus: 1
  hard_vocab_limit: 1
  use_all_vocab: 0
  unk_id: 0
  bos_id: 1
  eos_id: 2
  pad_id: -1
  unk_piece: <unk>
  bos_piece: <s>
  eos_piece: </s>
  pad_piece: <pad>
  unk_surface:  ⁇ 
  enable_differential_privacy: 0
  differential_privacy_noise_level: 0
  differential_privacy_clipping_threshold: 0
}
normalizer_spec {
  name: nmt_nfkc
  add_dummy_prefix: 1
  remove_extra_whitespaces: 1
  escape_whitespaces: 1
  normalization_rule_tsv: 
}
denormalizer_spec {}
trainer_interface.cc(351) LOG(INFO) SentenceIterator is not specified. Using MultiFileSentenceIterator.
trainer_interface.cc(183) LOG(INFO) Loading corpus: ./corpus/kannada_sentence_corpus_CulturaX_1GB.txt
trainer_interface.cc(378) LOG(WARNING) Found too long line (5325 > 4192).
trainer_interface.cc(380) LOG(WARNING) Too long lines are skipped in the training.
trainer_interface.cc(381) LOG(WARNING) The maximum length can be changed with --max_sentence_length=<size> flag.
trainer_interface.cc(145) LOG(INFO) Loaded 1000000 lines
trainer_interface.cc(122) LOG(WARNING) Too many sentences are loaded! (1940207), which may slow down training.
trainer_interface.cc(124) LOG(WARNING) Consider using --input_sentence_size=<size> and --shuffle_input_sentence=true.
trainer_interface.cc(127) LOG(WARNING) They allow to randomly sample <size> sentences from the entire corpus.
trainer_interface.cc(407) LOG(INFO) Loaded all 1940207 sentences
trainer_interface.cc(414) LOG(INFO) Skipped 8673 too long sentences.
trainer_interface.cc(423) LOG(INFO) Adding meta_piece: <unk>
trainer_interface.cc(423) LOG(INFO) Adding meta_piece: <s>
trainer_interface.cc(423) LOG(INFO) Adding meta_piece: </s>
trainer_interface.cc(428) LOG(INFO) Normalizing sentences...
trainer_interface.cc(537) LOG(INFO) all chars count=393822030
trainer_interface.cc(548) LOG(INFO) Done: 100% characters are covered.
trainer_interface.cc(558) LOG(INFO) Alphabet size=2124
trainer_interface.cc(559) LOG(INFO) Final character coverage=1
trainer_interface.cc(591) LOG(INFO) Done! preprocessed 1940207 sentences.
unigram_model_trainer.cc(222) LOG(INFO) Making suffix array...
unigram_model_trainer.cc(226) LOG(INFO) Extracting frequent sub strings... node_num=200925660
unigram_model_trainer.cc(274) LOG(INFO) Initialized 1002124 seed sentencepieces
trainer_interface.cc(597) LOG(INFO) Tokenizing input sentences with whitespace: 1940207
trainer_interface.cc(608) LOG(INFO) Done! 3713093
unigram_model_trainer.cc(564) LOG(INFO) Using 3713093 sentences for EM training
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=0 size=550722 obj=15.8184 num_tokens=10386773 num_tokens/piece=18.8603
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=1 size=443212 obj=12.9611 num_tokens=10433021 num_tokens/piece=23.5396
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=0 size=332136 obj=12.9228 num_tokens=10521980 num_tokens/piece=31.6797
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=1 size=330631 obj=12.9094 num_tokens=10538956 num_tokens/piece=31.8753
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=0 size=247968 obj=12.9239 num_tokens=10680222 num_tokens/piece=43.071
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=1 size=247963 obj=12.9187 num_tokens=10681176 num_tokens/piece=43.0757
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=0 size=185972 obj=12.9616 num_tokens=10957015 num_tokens/piece=58.9176
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=1 size=185971 obj=12.9475 num_tokens=10957315 num_tokens/piece=58.9195
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=0 size=139478 obj=13.049 num_tokens=11354131 num_tokens/piece=81.4045
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=1 size=139478 obj=13.0203 num_tokens=11354204 num_tokens/piece=81.405
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=0 size=104608 obj=13.1821 num_tokens=11749490 num_tokens/piece=112.319
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=1 size=104608 obj=13.1414 num_tokens=11749826 num_tokens/piece=112.322
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=0 size=78456 obj=13.3531 num_tokens=12193507 num_tokens/piece=155.418
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=1 size=78456 obj=13.3037 num_tokens=12194011 num_tokens/piece=155.425
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=0 size=58842 obj=13.5614 num_tokens=12678480 num_tokens/piece=215.467
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=1 size=58842 obj=13.5046 num_tokens=12678869 num_tokens/piece=215.473
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=0 size=44131 obj=13.804 num_tokens=13206780 num_tokens/piece=299.263
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=1 size=44131 obj=13.7397 num_tokens=13207147 num_tokens/piece=299.271
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=0 size=33098 obj=14.0896 num_tokens=13776107 num_tokens/piece=416.222
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=1 size=33098 obj=14.0164 num_tokens=13776947 num_tokens/piece=416.247
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=0 size=24823 obj=14.4245 num_tokens=14395856 num_tokens/piece=579.94
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=1 size=24823 obj=14.3384 num_tokens=14396654 num_tokens/piece=579.972
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=0 size=22000 obj=14.5207 num_tokens=14672567 num_tokens/piece=666.935
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=1 size=22000 obj=14.4833 num_tokens=14672937 num_tokens/piece=666.952
trainer_interface.cc(686) LOG(INFO) Saving model: kannada_sp.model
trainer_interface.cc(698) LOG(INFO) Saving vocabs: kannada_sp.vocab
Total time taken to train the model: 4.55 minutes
```

kannada_sentence_corpus_CulturaX_1-3M.txt
`python3 ./scripts/tokenisation/train.py --input-file "./corpus/kannada_sentence_corpus_CulturaX_1-3M.txt"`
```
sentencepiece_trainer.cc(77) LOG(INFO) Starts training with : 
trainer_spec {
  input: ./corpus/kannada_sentence_corpus_CulturaX_1-3M.txt
  input_format: 
  model_prefix: kannada_sp
  model_type: UNIGRAM
  vocab_size: 20000
  self_test_sample_size: 0
  character_coverage: 1
  input_sentence_size: 0
  shuffle_input_sentence: 1
  seed_sentencepiece_size: 1000000
  shrinking_factor: 0.75
  max_sentence_length: 4192
  num_threads: 16
  num_sub_iterations: 2
  max_sentencepiece_length: 16
  split_by_unicode_script: 1
  split_by_number: 1
  split_by_whitespace: 1
  split_digits: 0
  pretokenization_delimiter: 
  treat_whitespace_as_suffix: 0
  allow_whitespace_only_pieces: 0
  required_chars: 
  byte_fallback: 0
  vocabulary_output_piece_score: 1
  train_extremely_large_corpus: 1
  hard_vocab_limit: 1
  use_all_vocab: 0
  unk_id: 0
  bos_id: 1
  eos_id: 2
  pad_id: -1
  unk_piece: <unk>
  bos_piece: <s>
  eos_piece: </s>
  pad_piece: <pad>
  unk_surface:  ⁇ 
  enable_differential_privacy: 0
  differential_privacy_noise_level: 0
  differential_privacy_clipping_threshold: 0
}
normalizer_spec {
  name: nmt_nfkc
  add_dummy_prefix: 1
  remove_extra_whitespaces: 1
  escape_whitespaces: 1
  normalization_rule_tsv: 
}
denormalizer_spec {}
trainer_interface.cc(351) LOG(INFO) SentenceIterator is not specified. Using MultiFileSentenceIterator.
trainer_interface.cc(183) LOG(INFO) Loading corpus: ./corpus/kannada_sentence_corpus_CulturaX_1-3M.txt
trainer_interface.cc(378) LOG(WARNING) Found too long line (5325 > 4192).
trainer_interface.cc(380) LOG(WARNING) Too long lines are skipped in the training.
trainer_interface.cc(381) LOG(WARNING) The maximum length can be changed with --max_sentence_length=<size> flag.
trainer_interface.cc(145) LOG(INFO) Loaded 1000000 lines
trainer_interface.cc(145) LOG(INFO) Loaded 2000000 lines
trainer_interface.cc(145) LOG(INFO) Loaded 3000000 lines
trainer_interface.cc(145) LOG(INFO) Loaded 4000000 lines
trainer_interface.cc(145) LOG(INFO) Loaded 5000000 lines
trainer_interface.cc(145) LOG(INFO) Loaded 6000000 lines
trainer_interface.cc(145) LOG(INFO) Loaded 7000000 lines
trainer_interface.cc(145) LOG(INFO) Loaded 8000000 lines
trainer_interface.cc(145) LOG(INFO) Loaded 9000000 lines
trainer_interface.cc(145) LOG(INFO) Loaded 10000000 lines
trainer_interface.cc(145) LOG(INFO) Loaded 11000000 lines
trainer_interface.cc(145) LOG(INFO) Loaded 12000000 lines
trainer_interface.cc(145) LOG(INFO) Loaded 13000000 lines
trainer_interface.cc(145) LOG(INFO) Loaded 14000000 lines
trainer_interface.cc(145) LOG(INFO) Loaded 15000000 lines
trainer_interface.cc(145) LOG(INFO) Loaded 16000000 lines
trainer_interface.cc(145) LOG(INFO) Loaded 17000000 lines
trainer_interface.cc(145) LOG(INFO) Loaded 18000000 lines
trainer_interface.cc(145) LOG(INFO) Loaded 19000000 lines
trainer_interface.cc(145) LOG(INFO) Loaded 20000000 lines
trainer_interface.cc(122) LOG(WARNING) Too many sentences are loaded! (20151546), which may slow down training.
trainer_interface.cc(124) LOG(WARNING) Consider using --input_sentence_size=<size> and --shuffle_input_sentence=true.
trainer_interface.cc(127) LOG(WARNING) They allow to randomly sample <size> sentences from the entire corpus.
trainer_interface.cc(407) LOG(INFO) Loaded all 20151546 sentences
trainer_interface.cc(414) LOG(INFO) Skipped 77142 too long sentences.
trainer_interface.cc(423) LOG(INFO) Adding meta_piece: <unk>
trainer_interface.cc(423) LOG(INFO) Adding meta_piece: <s>
trainer_interface.cc(423) LOG(INFO) Adding meta_piece: </s>
trainer_interface.cc(428) LOG(INFO) Normalizing sentences...
trainer_interface.cc(537) LOG(INFO) all chars count=3739451208
trainer_interface.cc(548) LOG(INFO) Done: 100% characters are covered.
trainer_interface.cc(558) LOG(INFO) Alphabet size=6077
trainer_interface.cc(559) LOG(INFO) Final character coverage=1
trainer_interface.cc(591) LOG(INFO) Done! preprocessed 20150710 sentences.
unigram_model_trainer.cc(222) LOG(INFO) Making suffix array...
unigram_model_trainer.cc(226) LOG(INFO) Extracting frequent sub strings... node_num=2085776901
unigram_model_trainer.cc(274) LOG(INFO) Initialized 1006077 seed sentencepieces
trainer_interface.cc(597) LOG(INFO) Tokenizing input sentences with whitespace: 20150710
trainer_interface.cc(608) LOG(INFO) Done! 17233123
unigram_model_trainer.cc(564) LOG(INFO) Using 17233123 sentences for EM training
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=0 size=765758 obj=15.8375 num_tokens=55805720 num_tokens/piece=72.8764
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=1 size=607834 obj=12.9776 num_tokens=56129091 num_tokens/piece=92.3428
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=0 size=455727 obj=12.9348 num_tokens=56440105 num_tokens/piece=123.846
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=1 size=454603 obj=12.9258 num_tokens=56601106 num_tokens/piece=124.507
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=0 size=340952 obj=12.9274 num_tokens=56906830 num_tokens/piece=166.906
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=1 size=340950 obj=12.9257 num_tokens=56969745 num_tokens/piece=167.091
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=0 size=255712 obj=12.9354 num_tokens=57658711 num_tokens/piece=225.483
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=1 size=255712 obj=12.9332 num_tokens=57659361 num_tokens/piece=225.486
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=0 size=191784 obj=12.9628 num_tokens=58966601 num_tokens/piece=307.464
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=1 size=191784 obj=12.9531 num_tokens=58966121 num_tokens/piece=307.461
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=0 size=143838 obj=13.034 num_tokens=61282138 num_tokens/piece=426.05
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=1 size=143838 obj=13.01 num_tokens=61282515 num_tokens/piece=426.052
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=0 size=107878 obj=13.1659 num_tokens=63558543 num_tokens/piece=589.171
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=1 size=107878 obj=13.1249 num_tokens=63561565 num_tokens/piece=589.199
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=0 size=80908 obj=13.3457 num_tokens=66073613 num_tokens/piece=816.651
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=1 size=80908 obj=13.2928 num_tokens=66077397 num_tokens/piece=816.698
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=0 size=60681 obj=13.5696 num_tokens=68869808 num_tokens/piece=1134.95
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=1 size=60681 obj=13.5074 num_tokens=68872460 num_tokens/piece=1134.99
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=0 size=45510 obj=13.8421 num_tokens=72068994 num_tokens/piece=1583.59
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=1 size=45510 obj=13.7695 num_tokens=72074148 num_tokens/piece=1583.7
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=0 size=34132 obj=14.1841 num_tokens=75355233 num_tokens/piece=2207.76
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=1 size=34132 obj=14.0963 num_tokens=75361279 num_tokens/piece=2207.94
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=0 size=25599 obj=14.6153 num_tokens=79268257 num_tokens/piece=3096.54
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=1 size=25599 obj=14.5054 num_tokens=79272108 num_tokens/piece=3096.69
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=0 size=22000 obj=14.8148 num_tokens=81393422 num_tokens/piece=3699.7
unigram_model_trainer.cc(580) LOG(INFO) EM sub_iter=1 size=22000 obj=14.7519 num_tokens=81394919 num_tokens/piece=3699.77
trainer_interface.cc(686) LOG(INFO) Saving model: kannada_sp.model
trainer_interface.cc(698) LOG(INFO) Saving vocabs: kannada_sp.vocab
Total time taken to train the model: 34.06 minutes
```



(LLamaK-venv) adithya@llm-testing:~/k-llama/LLama-K/scripts/tokenisation$ python3 test.py 
1491
1221
1279
(LLamaK-venv) adithya@llm-testing:~/k-llama/LLama-K/scripts/tokenisation$ python3 test_edit.py 
../../models/kannada_sp_small.model 34824
../../models/kannada_sp_medium.model 31863
../../models/kannada_sp.model 33311
(LLamaK-venv) adithya@llm-testing:~/k-llama/LLama-K/scripts/tokenisation$ python3 test_edit.py 
../../models/kannada_sp_small.model 106574
../../models/kannada_sp_medium.model 96105
../../models/kannada_sp.model 99915



Medium : 
LLaMA tokenizer n_tokens=920109
kannada LLaMA tokenizer n_tokens=127329

Small :
LLaMA tokenizer n_tokens=920109
kannada LLaMA tokenizer n_tokens=137682