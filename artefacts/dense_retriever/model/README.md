---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:886
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-mpnet-base-v2
widget:
- source_sentence: What happens if an organisation fails to comply with its voluntary
    undertaking?
  sentences:
  - 'Powers of investigation

    To avoid doubt, despite subsection (3)(ca), the Commission may conduct or resume
    an investigation under this section at any time if an organisation or a person
    fails to comply with a voluntary undertaking given by the organisation or person
    under section 48L(1) in relation to any matter.'
  - 'Appeal from direction or decision of Commission

    A direction or decision of an Appeal Committee on an appeal has the same effect,
    and may be enforced in the same manner, as a direction or decision of the Commission,
    except that there is to be no application for further reconsideration under section
    48N and no further appeal under this section from the direction or decision of
    the Appeal Committee.'
  - 'Voluntary undertakings

    Where an organisation or a person fails to comply with any undertaking in a voluntary
    undertaking —

    (a) the Commission may give the organisation or person concerned any direction
    that the Commission thinks fit in the circumstances to ensure the compliance of
    the organisation or person with that undertaking; and

    (b) section 48K(1), (3), (4), (5), (6) and (7) applies to the direction given
    under paragraph (a) as if the direction were given under section 48I.'
- source_sentence: Can PDPC set a payment deadline of 14 days for a penalty if circumstances
    are urgent?
  sentences:
  - 'Procedure for giving of directions and imposing of financial penalty

    Where the Commission imposes a financial penalty under section 48J(1) on an organisation
    or a person, the written notice issued by the Commission to the organisation or
    person must specify the date before which the financial penalty is to be paid,
    being a date not earlier than 28 days after the notice is issued.'
  - 'Duty to notify occurrence of notifiable data breach

    The notification under subsection (1) or (2) must contain, to the best of the
    knowledge and belief of the organisation at the time it notifies the Commission
    or affected individual (as the case may be), all the information that is prescribed
    for this purpose.'
  - 'Contact information

    Subject to section 48(2), a person must not send a specified message addressed
    to a Singapore telephone number unless —

    (a) the specified message includes clear and accurate information identifying
    the individual or organisation that sent or authorised the sending of the specified
    message;

    (b) the specified message includes clear and accurate information about how the
    recipient can readily contact that individual or organisation;

    (c) the specified message includes the information, and complies with the conditions,
    specified in the regulations, if any; and

    (d) the information included in the specified message in compliance with this
    section is reasonably likely to be valid for at least 30 days after the message
    is sent.

    Calling line identity not to be concealed'
- source_sentence: Do companies have to tell job applicants why they’re collecting
    personal data?
  sentences:
  - 'Application of Act

    Parts 3, 4, 5, 6 (except sections 24 and 25), 6A (except sections 26C(3)(a) and
    26E) and 6B do not impose any obligation on a data intermediary in respect of
    its processing of personal data on behalf of and for the purposes of another organisation
    pursuant to a contract which is evidenced or made in writing.'
  - 'Notification of purpose

    Despite subsection (3), an organisation must comply with subsection (5) on or
    before collecting, using or disclosing personal data about an individual for the
    purpose of or in relation to the organisation —

    (a) entering into an employment relationship with the individual or appointing
    the individual to any office; or

    (b) managing or terminating the employment relationship with or appointment of
    the individual.'
  - 'Reconsideration of directions or decisions

    There is to be no application for reconsideration of a decision made under subsection
    (6)(b).

    Right of private action'
- source_sentence: If a terminated number receives an undeliverable message, could
    it still have a Singapore link?
  sentences:
  - 'Defence for employee

    Subsection (1), (2) or (3) does not apply to an employee (Z) who, at the time
    the act was done or the conduct was engaged in, was an officer or a partner of
    Z’s employer and it is proved that —

    (a) Z knew or ought reasonably to have known that the telephone number is a Singapore
    telephone number listed in the relevant register; and

    (b) the specified message was sent with Z’s consent or connivance, or the sending
    of the specified message was attributable to any neglect on Z’s part.'
  - 'Unauthorised disclosure of personal data

    In this section, “applicable contravention” means a contravention of any of the
    following:

    (a) subsection (1);

    (b) section 48F(1);

    (c) section 7(1) or 8(1) of the Public Sector (Governance) Act 2018;

    (d) section 14A(1) or 14C(1) of the Monetary Authority of Singapore Act 1970.

    Improper use of personal data'
  - 'Interpretation of this Part

    In this Part, an applicable message has a Singapore link in any of the following
    circumstances:

    (a) the message originates in Singapore;

    (b) the sender of the message —

    (i) where the sender is an individual — is physically present in Singapore when
    the message is sent; or

    (ii) in any other case —

    (A) is formed or recognised under the law of Singapore; or

    (B) has an office or a place of business in Singapore;

    (c) the telephone, mobile telephone or other device that is used to access the
    message is located in Singapore;

    (d) the recipient of the message —

    (i) where the recipient is an individual — is physically present in Singapore
    when the message is accessed; or

    (ii) in any other case — carries on business or activities in Singapore when the
    message is accessed;

    (e) if the message cannot be delivered because the telephone number to which the
    message is sent has ceased to exist (assuming that the telephone number existed),
    it is reasonably likely that the message would have been accessed using a telephone,
    mobile telephone or other device located in Singapore.'
- source_sentence: If a breach happened before 1 February 2021, must it be assessed
    under the PDPA?
  sentences:
  - 'Delegation

    In exercising any of the powers of enforcement under this Act, an authorised officer
    must on demand produce to the person against whom he or she is acting the authority
    issued to him or her by the Commission.

    Conduct of proceedings'
  - 'Duty to conduct assessment of data breach

    This section applies to a data breach that occurs on or after 1 February 2021.'
  - 'Offences and penalties

    A person guilty of an offence under subsection (1) shall be liable on conviction
    to a fine not exceeding $5,000 or to imprisonment for a term not exceeding 12
    months or to both.'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-mpnet-base-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) <!-- at revision e8c3b32edf5434bc2275fc9bab85f82640a19130 -->
- **Maximum Sequence Length:** 384 tokens
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
  (0): Transformer({'max_seq_length': 384, 'do_lower_case': False, 'architecture': 'MPNetModel'})
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
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
    'If a breach happened before 1 February 2021, must it be assessed under the PDPA?',
    'Duty to conduct assessment of data breach\nThis section applies to a data breach that occurs on or after 1 February 2021.',
    'Offences and penalties\nA person guilty of an offence under subsection (1) shall be liable on conviction to a fine not exceeding $5,000 or to imprisonment for a term not exceeding 12 months or to both.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[ 1.0000,  0.9018, -0.0146],
#         [ 0.9018,  1.0000,  0.0210],
#         [-0.0146,  0.0210,  1.0000]])
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

* Size: 886 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, <code>sentence_2</code>, and <code>label</code>
* Approximate statistics based on the first 886 samples:
  |         | sentence_0                                                                         | sentence_1                                                                          | sentence_2                                                                           | label                                                         |
  |:--------|:-----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                             | string                                                                              | string                                                                               | float                                                         |
  | details | <ul><li>min: 10 tokens</li><li>mean: 17.69 tokens</li><li>max: 37 tokens</li></ul> | <ul><li>min: 15 tokens</li><li>mean: 95.74 tokens</li><li>max: 384 tokens</li></ul> | <ul><li>min: 15 tokens</li><li>mean: 160.93 tokens</li><li>max: 384 tokens</li></ul> | <ul><li>min: 1.0</li><li>mean: 1.0</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                 | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | sentence_2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | label            |
  |:-------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>What powers does the Appeal Committee have when hearing an appeal?</code>            | <code>Appeal from direction or decision of Commission<br>An Appeal Committee hearing an appeal may confirm, vary or set aside the direction or decision which is the subject of the appeal and, in particular, may —<br>(a) remit the matter to the Commission;<br>(b) impose or revoke, or vary the amount of, a financial penalty;<br>(c) give any direction, or take any other step, that the Commission could itself have given or taken; or<br>(d) make any other direction or decision that the Commission could itself have made.</code>   | <code>Interpretation<br>In this Act, unless the context otherwise requires —<br>“advisory committee” means an advisory committee appointed under section 7;<br>“Appeal Committee” means a Data Protection Appeal Committee constituted under section 48P(4), read with the Seventh Schedule;<br>“Appeal Panel” means the Data Protection Appeal Panel established by section 48P(1);<br>“authorised officer”, in relation to the exercise of any power or performance of any function or duty under any provision of this Act, means a person to whom the exercise of that power or performance of that function or duty under that provision has been delegated under section 38 of the Info communications Media Development Authority Act 2016;<br>“Authority” means the Info communications Media Development Authority established by section 3 of the Info communications Media Development Authority Act 2016;<br>“benefit plan” means an insurance policy, a pension plan, an annuity, a provident fund plan or other similar plan;<br>“business” includes the...</code> | <code>1.0</code> |
  | <code>Can PDPC order destruction of unlawfully obtained data?</code>                       | <code>Directions for non compliance<br>Without limiting subsection (1), the Commission may, if it thinks fit in the circumstances to ensure compliance with any provision of Part 3, 4, 5, 6, 6A or 6B, give an organisation all or any of the following directions:<br>(a) to stop collecting, using or disclosing personal data in contravention of this Act;<br>(b) to destroy personal data collected in contravention of this Act;<br>(c) to comply with any direction of the Commission under section 48H(2).<br>Financial penalties</code> | <code>Meaning of “specified message”<br>Subject to subsection (5), for the purposes of this Part, a specified message is a message where, having regard to the following, it would be concluded that the purpose, or one of the purposes, of the message is an applicable purpose:<br>(a) the content of the message;<br>(b) the presentational aspects of the message;<br>(c) the content that can be obtained using the numbers, URLs or contact information (if any) mentioned in the message;<br>(d) if the telephone number from which the message is made is disclosed to the recipient (whether by calling line identity or otherwise), the content (if any) that can be obtained by calling that number.</code>                                                                                                                                                                                                                                                                                                                                                          | <code>1.0</code> |
  | <code>Does a company have to keep the preserved copy forever after refusing access?</code> | <code>Preservation of copies of personal data<br>Where —<br>(a) an individual, on or after 1 February 2021, makes a request under section 21(1)(a) to an organisation to provide personal data about the individual that is in the possession or under the control of the organisation; and<br>(b) the organisation refuses to provide that personal data,<br>the organisation must preserve, for not less than the prescribed period, a copy of the personal data concerned.</code>                                                              | <code>Unauthorised disclosure of personal data<br>If —<br>(a) an individual discloses, or the individual’s conduct causes disclosure of, personal data in the possession or under the control of an organisation or a public agency to another person;<br>(b) the disclosure is not authorised by the organisation or public agency, as the case may be; and<br>(c) the individual does so —<br>(i) knowing that the disclosure is not authorised by the organisation or public agency, as the case may be; or<br>(ii) reckless as to whether the disclosure is or is not authorised by the organisation or public agency, as the case may be,<br>the individual shall be guilty of an offence and shall be liable on conviction to a fine not exceeding $5,000 or to imprisonment for a term not exceeding 2 years or to both.</code>                                                                                                                                                                                                                                           | <code>1.0</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 50
- `fp16`: True
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
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
- `num_train_epochs`: 50
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
- `bf16`: False
- `fp16`: True
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
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
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
- `hub_revision`: None
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
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch   | Step | Training Loss |
|:-------:|:----:|:-------------:|
| 8.9286  | 500  | 0.1811        |
| 17.8571 | 1000 | 0.1091        |
| 26.7857 | 1500 | 0.0988        |
| 35.7143 | 2000 | 0.1017        |
| 44.6429 | 2500 | 0.0945        |


### Framework Versions
- Python: 3.11.13
- Sentence Transformers: 5.1.1
- Transformers: 4.57.0
- PyTorch: 2.8.0
- Accelerate: 1.11.0
- Datasets: 4.3.0
- Tokenizers: 0.22.1

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

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
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