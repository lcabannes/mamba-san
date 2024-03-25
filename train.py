import torch
import argparse
import sys
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer, TrainingArguments
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.hf import load_config_hf
from datasets import Dataset, load_dataset, DatasetDict
from trainer.mamba_trainer import MambaTrainer
from trainer.MambaConfig import MambaConfig
from trainer.data import DataCollatorForChatDataset
import wandb


def run(args):
    """

    :param args:
    model_name, type=str, default="MambaSan":
    load_model, type=str, default=None): to train starting from a model checkpoint or another pretrained model
    load_config, type=str, default=None) : to instantiate a new model with from a certain config,
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if args.load_model is not None:
        model = MambaLMHeadModel.from_pretrained(f"{args.load_model}", dtype=torch.bfloat16, device=device)
        model.config = MambaConfig(vars(model.config))
        print(f"model {args.load_model} loaded")
    else:
        tokenizer.pad_token = tokenizer.eos_token
        if args.load_config is not None:
            config_data = load_config_hf(args.load_config)
        else:
            config_data = load_config_hf('state-spaces/mamba-130m')
            config_data["d_model"] = args.d_model
            config_data["n_layer"] = args.n_layer
        config_data["vocab_size"] = tokenizer.vocab_size
        config = MambaConfig(config_data)
        model = MambaLMHeadModel(config, dtype=torch.bfloat16, device=device)
    tokenizer.pad_token = tokenizer.eos_token

    if args.verbosity >= 1:
        model_size = sum(t.numel() for t in model.parameters())
        print(f"model size: {model_size / 1000 ** 2:.1f}M parameters")

    dataset = load_dataset(args.data_path, lang=args.lang,
                           split="train",
                           streaming=True,
                           trust_remote_code=True)

    if args.report_to == "wandb":
        wandb.init(project=args.model_name, name=args.output_dir)

    dataset = dataset.skip(args.n_dataset_skip)

    def group_texts(examples):
        block_size = args.context_length
        # Concatenate all texts.
        tokenized_examples = tokenizer(examples.pop("text"))["input_ids"]
        concatenated_examples = sum([ex + [tokenizer.eos_token_id] for ex in tokenized_examples], [])
        total_length = len(concatenated_examples)
        # Split by chunks of block_size.
        result = {
            "input_ids": [concatenated_examples[i: i + block_size] for i in range(0, total_length, block_size)]
        }
        return result


    if args.streaming:
        if 0 < args.eval_size < 1:
            args.eval_size = args.eval_size * args.dataset_size
        args.eval_size = int(args.eval_size)
        eval_set = Dataset.from_list(list(dataset.take(max(1, args.eval_size))))
        train_set = dataset.skip(args.eval_size)

    else:
        dataset = Dataset.from_list(list(dataset.take(args.dataset_size)))
        dataset = dataset.skip(args.dataset_size)
        if args.eval_size > 0:
            raw_datasets = dataset.train_test_split(test_size=args.eval_size, shuffle=True)
        else:
            raw_datasets = DatasetDict({"train": dataset, "test": Dataset.from_list(list(dataset.take(1)))})
        train_set = raw_datasets.pop("train")
        eval_set = raw_datasets.pop("test")


    if args.verbosity >= 1:
        print(train_set)
        print(eval_set)
        if not args.streaming:
            longest_text = max([len(ex) for ex in train_set["text"]])
            average_length = sum([len(ex) for ex in train_set["text"]]) / train_set.num_rows
            print(f"longest_length in dataset: {longest_text}")
            print(f"average example length {average_length}")


    train_set = train_set.map(
        group_texts,
        batched=True,
        remove_columns=train_set.column_names,
        )
    eval_set = eval_set.map(
        group_texts,
        batched=True,
        remove_columns=eval_set.column_names,
        )

    if args.streaming:
        train_set = train_set.with_format("torch")
    else:
        train_set.set_format(type="torch")
    eval_set.set_format(type="torch")
    data_collator = DataCollatorForChatDataset(tokenizer)

    if args.verbosity >= 1:
        print(train_set)
        print(eval_set)
        out = data_collator([eval_set[0]])
        for key in out:
            print(f"{key} shape: {out[key].shape}")

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps" if args.eval_steps > 0 and args.eval_size > 0 else "no",
        logging_dir=f"{args.output_dir}/logs",
        eval_steps=args.eval_steps,
        lr_scheduler_type="constant",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        save_strategy="steps",
        bf16=True,
        push_to_hub=False,
        optim=args.optim,
        report_to=[args.report_to] if args.report_to else [],
        run_name=args.output_dir,
        weight_decay=0.1,
        label_names=["labels"],
        logging_first_step=True,
        max_steps=args.dataset_size if args.streaming else -1
    )

    trainer = MambaTrainer(
        model=model,
        eval_dataset=eval_set,
        train_dataset=train_set,
        tokenizer=tokenizer,
        args=train_args,
        data_collator=data_collator,
    )
    if args.restart_from is not None:
        trainer.train(args.restart_from)
    print(f"starting training")
    trainer.train()
    model.save_pretrained(save_directory=args.output_dir)
    tokenizer.save_pretrained(save_directory=args.output_dir)
    print(f"model {args.model_name} and tokenizer saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="MambaSan")
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--load_config", type=str, default=None)
    parser.add_argument("--restart_from", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default="abeja/gpt-neox-japanese-2.7b")
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--context_length", type=int, default=2048)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="out")
    parser.add_argument("--report_to", type=str, default=None)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--data_path", type=str, default="cc100")
    parser.add_argument("--lang", type=str, default="ja")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_layer", type=int, default=24)
    parser.add_argument("--verbosity", type=int, default=0)
    parser.add_argument("--pad_vocab_size_multiple", type=int, default=8)
    parser.add_argument("--n_dataset_skip", type=int, default=0)
    parser.add_argument("--dataset_size", type=int, default=10000)
    parser.add_argument("--eval_size", type=float, default=100)
    parser.add_argument("--logging_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--streaming", type=bool, default=True)
    args = parser.parse_args()

    run(args)

