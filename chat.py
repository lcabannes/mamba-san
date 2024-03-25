import argparse

import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer
import time

CHAT_MODE = True
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer if args.tokenizer is not None else args.load_model)
    print(f"tokenizer {args.tokenizer if args.tokenizer is not None else args.load_model} loaded")
    model = MambaLMHeadModel.from_pretrained(f"{args.load_model}", dtype=torch.bfloat16, device=device)
    print(f"model {args.load_model} loaded")

    while True:
        text = input("user: ").encode('utf-8', 'replace').decode()
        if args.generation_type == "chat":
            input_ids = tokenizer.apply_chat_template([{"role":"user", "content":text}], return_tensors="pt")
        else:
            input_ids = tokenizer.encode(text, return_tensors="pt")
        start_time = time.time()
        out = model.generate(input_ids.to(device),
                             cg=False,
                             max_length=2048,
                             temperature=1,
                             eos_token_id=tokenizer.eos_token_id,
                             top_p=0.7,
                             top_k=10,
                             enable_timing=True
                             )
        end_time = time.time()

        print()
        print(tokenizer.decode(out[0], ignore_special_tokens=True))
        print(f"processed {len(out[0])} tokens in {(end_time - start_time)}s: {len(out[0]) / (end_time - start_time)} tokens/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_model", type=str, default="loiccabannes/MambaSan-130m-instruct")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--generation_type", type=str, default="chat")
    run(parser.parse_args())