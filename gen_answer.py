import datasets
from fastchat.model.model_adapter import load_model
import os
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--ans-path",
        type=str,
        help="directory to store then answer.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    args = parser.parse_args()

    model, tokenizer = load_model(
            args.model_path,
            device="cuda",
            num_gpus=args.num_gpus_per_model,
            max_gpu_memory=args.max_gpu_memory,
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )


    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    for example in eval_set:
        example["output"] = model.generate(example["instruction"])

    ans_file = args.ans_path
    with open(ans_file, 'w') as file:
        json.dump(eval_set, file)
    file.close()