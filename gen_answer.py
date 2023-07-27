import datasets
import torch
from fastchat.model import load_model, get_conversation_template
import argparse
import json
from tqdm import tqdm

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
        "--model-id",
        type=str,
        default="airoboros-v1",
        required=True
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=256,
        help="The max number of new tokens",
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

    temperature = 0.7
    do_sample = True
    torch.manual_seed(0)

    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    for example in tqdm(eval_set):
        conv = get_conversation_template(args.model_id)
        conv.append_message(conv.roles[0], example["instruction"])
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer([prompt]).input_ids

        output_ids = model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=args.max_new_token,
        )
        if model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(input_ids[0]):]
        output = tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
        )

        if conv.stop_str:
            output = output[: output.find(conv.stop_str)]
        output = output.strip()
        if conv.name == "xgen" and output.startswith("Assistant:"):
            output = output.replace("Assistant:", "", 1).strip()

        conv.messages[-1][-1] = output
        example["output"] = output

    ans_file = args.ans_path
    with open(ans_file, 'w') as file:
        json.dump(eval_set, file)
    file.close()