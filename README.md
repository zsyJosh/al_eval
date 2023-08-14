# AlpacaEval : An Automatic Evaluator for Instruction-following Language Models

Evaluation of instruction-following models (e.g., ChatGPT) typically requires human interactions. This is
time-consuming, expensive, and hard to replicate. AlpacaEval in an LLM-based automatic evaluation that is fast, cheap,
replicable, and validated against 20K human annotations.
It is particularly useful for model development.
Although we improved over prior automatic evaluation pipelines, there are still fundamental [limitations](#limitations) like the preference for longer outputs.
AlpacaEval provides the following:


# Quick Start

This evaluation method works with two steps: 1) Generate the model's answer to a specific instruction dataset (say, alpaca_eval dataset). 2) LLM-judge compares between the answers from
two models and output the winning rate of *model_outputs* over *reference_outputs*.

## Install
To install, run the following command

```bash
python setup.py install
```


## Generate answers
First, you need to generate the model's answer as follows:

```bash
python gen_answer.py --model-path <the_path_of_model> \
--ans-path  <path_to_generated_answer> \
--model-id <the_name_of_model>
```

## Specify Openai account

```bash
export OPENAI_API_TYPE=
export OPENAI_API_BASE=
export OPENAI_API_VERSION=
export OPENAI_API_KEYS=
export OPENAI_ENGINE=
```


## Evaluating a model
Then you can evaluate the model as follows:

```bash
alpaca_eval --model_outputs <path_to_generated_answer> \
--annotators_config <llm-judge_config> \
-- reference_outputs <path_to_referenced_answer>
```

This will print the leaderboard to the console, and save both the leaderboard and the annotations to the same directory as the `model_outputs` file. Important parameters are the following:

- **model_outputs** : A path to a json file for the outputs of the model. The answer contains the keys `instruction` and `output`.
- **annotators_config**: This is the annotator to use. Configs can be found in `/src/alpaca_eval/evaluators_configs`, which can be referenced and further modified (Indicated below). Our annotator is `/src/alpaca_eval/evaluators_configs/chat_eval/configs.yaml` 
- **reference_outputs**:  The outputs of the reference model. Same format as `model_outputs`. Some existing reference_outputs can be found in `/results`
- **output_path**: Path for saving annotations and leaderboard.


## Making a new evaluator

<details>
  <summary><code>>>> alpaca_eval analyze_evaluators -- --help</code></summary>

```
NAME
    alpaca_eval analyze_evaluators - Analyze an evaluator and populates the evaluators leaderboard (agreement with human, speed, price,...).

SYNOPSIS
    alpaca_eval analyze_evaluators <flags>

DESCRIPTION
    Analyze an evaluator (agreement with human, speed, price,...).

FLAGS
    --annotators_config=ANNOTATORS_CONFIG
        Type: Union
        Default: 'alpaca_eval_gpt4'
        The path the (or list of dict of) the annotator's config file.
    -A, --Annotator=ANNOTATOR
        Default: <class 'alpaca_eval.annotators.pairwise_evaluator.PairwiseAn...
        The annotator class to use.
    --analyzer_kwargs=ANALYZER_KWARGS
        Type: Optional[]
        Default: None
        Additional arguments to pass to the analyzer.
    -p, --precomputed_leaderboard=PRECOMPUTED_LEADERBOARD
        Type: Union
        Default: PosixPath('/Users/yanndubois/Desktop/GitHub/alpaca_eval/src/...
        The precomputed (meta)leaderboard of annotators or a path to it (json, csv, or tsv).
    --is_save_leaderboard=IS_SAVE_LEADERBOARD
        Type: bool
        Default: False
        Whether to save the leaderboard (ie analyzed results).
    --is_return_instead_of_print=IS_RETURN_INSTEAD_OF_PRINT
        Type: bool
        Default: False
        Whether to return the leaderboard (ie analyzed results). If True, it will not print the results.
    --is_overwrite_leaderboard=IS_OVERWRITE_LEADERBOARD
        Type: bool
        Default: False
        Whether to overwrite the leaderboard if it already exists.
    -m, --max_instances=MAX_INSTANCES
        Type: Optional[Optional]
        Default: None
        The maximum number of instances to analyze.
    --is_single_annotator=IS_SINGLE_ANNOTATOR
        Type: bool
        Default: False
        Whether to analyze a single annotator. If True, will not be able to estimate the annotator's bias.
```

</details>

AlpacaEval provides a simple way of making new evaluators. All you need is to make a new `configs.yaml` configuration
file, which you will then pass
as `--annotators_config <path_to_config.yaml>` to `alpaca_eval`.
Here are some ways you can make a new evaluator:

- **Changing the prompt**: Write a new prompt in a text file and specify the path in `prompt_template` of the
  configuration file. Paths are relative to the configuration file.
- **Changing decoding parameters**: Specify the desired parameters in `completions_kwargs` in the configuration file. To
  see all available parameters refer to the docstrings of the corresponding
  function [in this file](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/decoders/__init__.py)
  specified by `fn_completions`
  in the configuration file.
- **Changing the model**: Specify the desired model in `model_name` and the corresponding
  prompt in `prompt_template`. If the model comes from another provider you
  will
  have
  to change `fn_completions` which maps to the corresponding function
  in [this file](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/decoders/__init__.py). We
  provide `fn_completions` functions to use models from OpenAI, Anthropic, Cohere, or HuggingFace. To
  install packages needed for
  all providers
  use `pip install alpaca_eval[all]`.

[//]: # (- **Using multiple annotators**: Specify a list of annotators in `annotators_config` in the configuration file. For an)

[//]: # (  example)

[//]: # (  see [alpaca_farm configuration]&#40;https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/evaluators_configs/alpaca_farm/configs.yaml&#41;.)

<details>
  <summary><b>Other parameters in the configuration file</b></b></summary>

The easiest is to check the docstrings
of [`SinglePairwiseAnnotator`](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/annotators/pairwise_evaluator.py#L537).
Here are some important ones:

```
Parameters
----------
prompt_template : path
    A prompt that will be given to `fn_prompter` or path to the prompts. Path is relative to
    `evaluators_configs/`

fn_completion_parser : callable or str
    Function in `completion_parsers.py` to use for parsing the completions into preferences. For each completion,
    the number of preferences should be equal to the batch_size if not we set all the preferences in that batch to
    NaN.

completion_parser_kwargs : dict
    Kwargs for fn_completion_parser.

fn_completions : callable or str
    Function in `decoders.py` to use for decoding the output.

completions_kwargs : dict
    kwargs for fn_completions. E.g. model_name, max_tokens, temperature, top_p, top_k, stop_seq.

is_randomize_output_order : bool
    Whether to randomize output_1, output_2 when formatting.

batch_size : int
    Number of examples that will be added in a single prompt.
```

</details>

Once you made the evaluator you can also analyze it and add it to the _evaluator's_ [leaderboard](#evaluators) using the
following command:

```bash
alpaca_eval analyze_evaluators --annotators_config '<path_to_config.yaml>'    
```

To estimate the bias and variance this evaluates every example with 4 seeds, i.e., 2.5K
evaluation.
If you want a cheaper evaluation you can use a single seed using `--is_single_annotator True` which will skip the
estimation of bias and variance.

# Citation

Please consider citing the repo if you used the automatic annotators, code, or results.

```
@misc{alpaca_eval,
  author = {Xuechen Li and Tianyi Zhang and Yann Dubois and Rohan Taori and Ishaan Gulrajani and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto },
  title = {AlpacaEval: An Automatic Evaluator of Instruction-following Models},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tatsu-lab/alpaca_eval}}
}
```

If you used our human annotation data, please also consider citing the [AlpacaFarm](https://arxiv.org/abs/2305.14387)
paper:

```
@misc{dubois2023alpacafarm,
  title={AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback}, 
  author={Yann Dubois and Xuechen Li and Rohan Taori and Tianyi Zhang and Ishaan Gulrajani and Jimmy Ba and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto},
  year={2023},
  eprint={2305.14387},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

If you use the AlpacaEval evaluation set, please cite each of the constituent
datasets: [self-instruct](https://github.com/yizhongw/self-instruct),
[open-assistant](https://huggingface.co/datasets/OpenAssistant/oasst1/viewer/OpenAssistant--oasst1/validation), [vicuna](https://lmsys.org/blog/2023-03-30-vicuna/), [koala](https://github.com/arnav-gudibande/koala-test-set), [hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf/viewer/Anthropic--hh-rlhf/test).
