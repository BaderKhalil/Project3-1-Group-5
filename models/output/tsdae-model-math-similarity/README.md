---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:4370
- loss:DenoisingAutoEncoderLoss
base_model: math-similarity/Bert-MLM_arXiv-MP-class_zbMath
widget:
- source_sentence: 'amy is trying to work out the distance between these two points:
    (2,-5) and (-7,1) she labels them like this: \begin {array} {cccc} x_ {1} & y_
    {1} & x_ {2} & y_ {2} (2, & -5) & (-7, & 1) \end {array} and then decides to use
    this formula: \sqrt { \left (x_ {2} -x_ {1} \right ^ {2} + \left (y_ {2} -y_ {1}
    \right ^ {2} } \sqrt { (-9) ^ {2} + (6) ^ {2} } what is the distance between the
    points? - \sqrt {45}'
  sentences:
  - Does not convert measurements to have the same units before calculating area or
    volume
  - Believes the square of a negative will also be negative
  - Believes you can subtract from inside brackets without expanding when solving
    an equation
- source_sentence: "tom and katie are discussing congruent shapes.\r tom says you\
    \ can reflect a shape and it will still be congruent to the original.\r \r katie\
    \ says you can rotate a shape and it will still be congruent to the original.\r\
    \ who is correct? neither is correct"
  sentences:
  - When asked for the mean of a list of data, gives the median
  - When solving a problem that requires an inverse operation (e.g. missing number
    problems), does the original operation
  - Does not understand that shapes are congruent if they have the same size and shape
- source_sentence: "what is the size of the marked angle? ! [a 360 degree protractor\
    \ with 2 red lines and a pink sector marking out a reflex angle being measured\
    \ by the protractor. the outer scale goes from 0 to 360 clockwise. the angle is\
    \ between one red line that is at 0 (outer scale) and 180 (inner scale) , clockwise\
    \ to the other red line that is half way between 210 and 220 (outer scale) and\
    \ halfway between 330 and 320 (inner scale) .\r () 335^ { \\circ }"
  sentences:
  - Thinks you need to just add a % sign to a decimal to convert to a percentage
  - Reads the wrong scale on the protractor when measuring reflex angles
  - Factorises difference of two squares with the same sign in both brackets
- source_sentence: which shape has rotational symmetry order 4 ? ! [trapezium] ()
  sentences:
  - Does not know how to find order of rotational symmetry
  - Assumed each part of a fraction diagram is worth 10%
  - Difficulty with translating a mathematical answer into a real world context
- source_sentence: 3 \times (-5) = 15
  sentences:
  - When multiplying with negative numbers, assumes any negative sign can be ignored
  - Makes an assumption about line segments being equal within a shape
  - Thinks terms in linear sequence are in direct proportion
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on math-similarity/Bert-MLM_arXiv-MP-class_zbMath

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [math-similarity/Bert-MLM_arXiv-MP-class_zbMath](https://huggingface.co/math-similarity/Bert-MLM_arXiv-MP-class_zbMath). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [math-similarity/Bert-MLM_arXiv-MP-class_zbMath](https://huggingface.co/math-similarity/Bert-MLM_arXiv-MP-class_zbMath) <!-- at revision 6e1701b8580a3ff9b85bce1cf3e072f64751dd59 -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    '3 \\times (-5) = 15',
    'When multiplying with negative numbers, assumes any negative sign can be ignored',
    'Makes an assumption about line segments being equal within a shape',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset


* Size: 4,370 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                         | sentence_1                                                                        |
  |:--------|:-----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                            |
  | details | <ul><li>min: 7 tokens</li><li>mean: 50.05 tokens</li><li>max: 283 tokens</li></ul> | <ul><li>min: 4 tokens</li><li>mean: 14.51 tokens</li><li>max: 38 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                 | sentence_1                                                                                                        |
  |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------|
  | <code>which of the following answers shows a correct expansion and simplification of the expression below?    (   (x-3) ^ {2} x^ {2} +9</code>                                                                                                              | <code>Believes they only need to multiply the first and last pairs of terms when expanding double brackets</code> |
  | <code>tom and katie want to share £2 between 4 people.   tom says you would calculate the amount each person gets by doing 4 \div 2    katie says you would calculate the amount each person gets by doing 2 \div 4    who do you agree with? only tom</code> | <code>Believes division is commutative </code>                                                                    |
  | <code>here are the first three terms of a sequence ! \begin {array} {llll} 98 & 92 & 86 & \ldots \end {array} () find an expression for the nth term of this sequence. -6 n</code>                                                                         | <code>Thinks terms in linear sequence are in direct proportion</code>                                             |
* Loss: [<code>DenoisingAutoEncoderLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#denoisingautoencoderloss)

### Training Hyperparameters
#### Non-Default Hyperparameters

- `num_train_epochs`: 1
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 8
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.9141 | 500  | 5.8594        |


### Framework Versions
- Python: 3.11.0
- Sentence Transformers: 3.3.1
- Transformers: 4.47.1
- PyTorch: 2.4.0+cpu
- Accelerate: 1.2.1
- Datasets: 3.2.0
- Tokenizers: 0.21.0

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### DenoisingAutoEncoderLoss
```bibtex
@inproceedings{wang-2021-TSDAE,
    title = "TSDAE: Using Transformer-based Sequential Denoising Auto-Encoderfor Unsupervised Sentence Embedding Learning",
    author = "Wang, Kexin and Reimers, Nils and Gurevych, Iryna",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    pages = "671--688",
    url = "https://arxiv.org/abs/2104.06979",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->