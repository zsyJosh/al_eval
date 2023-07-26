import datasets
from fastchat.model.model_adapter import load_model
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True)
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

    model_path = os.environ.get("MODEL_PATH", None)
    assert model_path is not None

    model, tokenizer = load_model(
            model_path,
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