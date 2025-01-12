---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:4370
- loss:DenoisingAutoEncoderLoss
base_model: sentence-transformers/all-MiniLM-L12-v2
widget:
- source_sentence: "solve the inequality:\r (- \\frac {1} {3} y \\leq -6 y \\geq 2"
  sentences:
  - Believes you find the frequency for a pie chart sector by calculating its angle
    divided by the total frequency
  - Believes dividing by a unit fraction is equivalent to dividing by its reciprocal
  - When collecting like terms, treats subtractions as if they are additions
- source_sentence: what is \frac {13} {40} written as a percentage? 26 %
  sentences:
  - Does not know how to find missing lengths in a composite shape
  - Uses more than 3 letters to describe an angle
  - Believes a fraction out of a number other than 100 represents a percentage
- source_sentence: 'which line gives the reflection of shape p onto shape q? ! [a
    coordinate grid with two right angled triangles drawn, labelled p and q. p has
    the coordinates: (2,5) (2,8) and (4,5) . q has the coordinates: (5,2) (8,2) and
    (5,4) .] () y=4.5'
  sentences:
  - Rounds down instead of up
  - Believes lines of reflection must be vertical or horizontal
  - Does not follow the arrows through a function machine, changes the order of the
    operations asked.
- source_sentence: "when h=5 \r which of the following pairs of statements is true?\
    \ \\begin {array} {l} 3 h^ {2} =225 (3 h) ^ {2} =225 \\end {array}"
  sentences:
  - 'Multiplies before applying a power '
  - Does not recognise the distributive property
  - Confuses square rooting and halving the number
- source_sentence: "tom is has correctly drawn the graph of the following function:\r\
    \ (\r y= \\left { \\begin {array} {c} \r 3 \\text { for } 0 \\leq x<q \r p \\\
    text { for } 1.5 \\leq x<4 \r 2.5 \\text { for } 4 \\leq x<7\r \\end {array} \\\
    right .\r \r \r what are the values of p and q ? ! [the graph is three separate\
    \ horizontal lines. it goes from (0,3) to (1.5, 3) , then (1.5, 2) to (4, 2) and\
    \ the last section is from (4, 2.5) to (7, 2.5) () \\begin {array} {c} p=1.5 q=2\
    \ \\end {array}"
  sentences:
  - When reading value from graph, reads from the wrong axes.
  - When reading integers on a number line, assumes each dash is 1
  - Does not know that a right angle is 90 degrees
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L12-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) <!-- at revision 364dd28d28dcd3359b537f3cf1f5348ba679da62 -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 384 dimensions
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
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
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
    'tom is has correctly drawn the graph of the following function:\r (\r y= \\left { \\begin {array} {c} \r 3 \\text { for } 0 \\leq x<q \r p \\text { for } 1.5 \\leq x<4 \r 2.5 \\text { for } 4 \\leq x<7\r \\end {array} \\right .\r \r \r what are the values of p and q ? ! [the graph is three separate horizontal lines. it goes from (0,3) to (1.5, 3) , then (1.5, 2) to (4, 2) and the last section is from (4, 2.5) to (7, 2.5) () \\begin {array} {c} p=1.5 q=2 \\end {array}',
    'When reading value from graph, reads from the wrong axes.',
    'When reading integers on a number line, assumes each dash is 1',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

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
  | details | <ul><li>min: 8 tokens</li><li>mean: 51.17 tokens</li><li>max: 251 tokens</li></ul> | <ul><li>min: 4 tokens</li><li>mean: 14.88 tokens</li><li>max: 39 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                 | sentence_1                                                                                                |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------|
  | <code>tom and katie are arguing about points on the line   [   y=2 x-3     tom says (5,7) lies on the line \boldsymbol {y} = \mathbf {2} \boldsymbol {x} - \mathbf {3}     katie says (-5,-7) lies on the line \boldsymbol {y} = \mathbf {2 x} - \mathbf {3}     who is correct? both tom and katie</code>                  | <code>Believes subtracting a positive number from a negative number makes the answer less negative</code> |
  | <code>which of the following correctly describes the marked angle? ! [an image of square abce with various triangles attached to it. the marked angle is between the lines ce and eg.] () fec</code>                                                                                                                       | <code>In 3 letter angle notation, gives a wider angle that contains the shaded angle</code>               |
  | <code>! [a straight line on squared paper. points p, q and r lie on this line. the leftmost end of the line is labelled p. if you travel right 4 squares and up 1 square you get to point q. if you then travel 8 squares right and 2 squares up from q you reach point r.] () what is the ratio of p q: p r ? 1: 4</code> | <code>May have estimated when using ratios with geometry</code>                                           |
* Loss: [<code>DenoisingAutoEncoderLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#denoisingautoencoderloss)

### Training Hyperparameters
#### Non-Default Hyperparameters

- `num_train_epochs`: 2
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
- `num_train_epochs`: 2
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
| 0.9141 | 500  | 5.6792        |
| 1.8282 | 1000 | 4.2975        |


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