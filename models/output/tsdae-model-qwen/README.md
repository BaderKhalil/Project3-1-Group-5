---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:4370
- loss:DenoisingAutoEncoderLoss
base_model: Qwen/Qwen2.5-0.5B-Instruct
widget:
- source_sentence: 0.2 \times 0.8= 1
  sentences:
  - Mixes up greater than and less than symbols
  - Adds instead of multiplies
  - Thinks numbers are a multiple of their index
- source_sentence: '! [a function machine with 4 rectangles in a row, joined by arrows
    pointing from left to right. the first rectangle on the left is empty and says
    input above it. the next rectangle has − 4 written inside it, the next rectangle
    has ✕ 4 written inside it and the final rectangle on the right has output written
    above it and 8m − 20 written inside it.] () what is the input of this function
    machine? 2 m-20'
  sentences:
  - Only multiplies second term in the expansion of a bracket
  - When calculating the range does not reorder the data to find the largest number
    minus the smallest number
  - Does not apply the inverse function to every term in an expression
- source_sentence: "which of the following is an example of a quadratic graph? ! [a\
    \ graph with two curves. one curve is in the positive x, positive y quadrant.\
    \ it starts at the top close to the y axis, as we move right it drops very quickly\
    \ towards the x axis then levels off to travel almost parallel to the x axis.\r\
    \ the other curve is in the negative x, negative y quadrant. it starts at the\
    \ top left close to the x axis, as we move right it drops very quickly towards\
    \ the y axis then becomes almost vertical to travel almost parallel to the y axis.\
    \ ()"
  sentences:
  - Does not recognise when to find a factor from a worded question
  - Confuses reciprocal and quadratic graphs
  - Does not understand equivalent fractions
- source_sentence: "! [a linear graph showing that 10 miles = £8. () the graph can\
    \ be used to work out how much kay's company pays her for travel.\r \r kay's company\
    \ paid her £ 80 \r \r how many miles did she travel? 80"
  sentences:
  - Believes direct proportion means a 1:1 ratio
  - When factorising, finds a factor that goes into only the first term of the expression
  - Thinks you multiply parallel sides to find the area of a trapezium
- source_sentence: 427 \times 6= 240+120+42
  sentences:
  - When multiplying multiples of ten and the answer requires an extra digit, leaves
    off that extra digit
  - 'Does not understand the meaning of the word commutative '
  - Believes an interior angle in a regular polygon can be found using 180(n+2)/n
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on Qwen/Qwen2.5-0.5B-Instruct

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct). It maps sentences & paragraphs to a 896-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) <!-- at revision 7ae557604adf67be50417f59c2c2f167def9a775 -->
- **Maximum Sequence Length:** 32768 tokens
- **Output Dimensionality:** 896 dimensions
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
  (0): Transformer({'max_seq_length': 32768, 'do_lower_case': False}) with Transformer model: Qwen2Model 
  (1): Pooling({'word_embedding_dimension': 896, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
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
    '427 \\times 6= 240+120+42',
    'When multiplying multiples of ten and the answer requires an extra digit, leaves off that extra digit',
    'Does not understand the meaning of the word commutative ',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 896]

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
  | details | <ul><li>min: 8 tokens</li><li>mean: 54.16 tokens</li><li>max: 295 tokens</li></ul> | <ul><li>min: 4 tokens</li><li>mean: 12.79 tokens</li><li>max: 37 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                             | sentence_1                                                                                               |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------|
  | <code>this is a table of values for 3 x+2 y=6    what should replace the star? \begin {tabular} { | l | c | c | c | c | c | }    \hline x & 0 & 1 & 2 & 3 & 4    \hline y & \color {gold} \bigstar &    \hline    \end {tabular} 1.5</code> | <code>Believes 0 multiplied by a number gives the number</code>                                          |
  | <code>factorise this expression, if possible:   (   p^ {2} +4 p (p+2) (p+2)</code>                                                                                                                                                       | <code>When factorising a quadratic without a non variable term, tries to double bracket factorise</code> |
  | <code>tom and katie are discussing sequences   tom says \boldsymbol {n} + \mathbf {1} ^ {3} would produce a linear sequence.   katie says 3- \frac {n} {3} would produce a linear sequence   who is correct? neither is correct</code>    | <code>Believes linear sequences cannot include fractions</code>                                          |
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
| 0.9141 | 500  | 1.4995        |


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