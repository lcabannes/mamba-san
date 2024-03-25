import torch
import transformers

from dataclasses import dataclass
from typing import Dict, Sequence
from tqdm import tqdm



@dataclass
class DataCollatorForLanguageModelling(object):
    """
    Collate examples for supervised pre-training.
    """

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # copy input ids for labels
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "input_ids"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        # shift labels to the right by one
        labels = labels[:, 1:]

        return dict(
            input_ids=input_ids,
            labels=labels,
        )


def preprocess(conversations: Sequence[Sequence[dict]], tokenizer: transformers.PreTrainedTokenizer, conversation_template: str, max_tokens: int) -> Dict:
    """
    Preprocess the data by tokenizing.
    """
    all_input_ids = []
    all_label_ids = []
    tokenizer.use_default_system_prompt = False

    print("Tokenizing dataset...")
    for conv in tqdm(conversations):
        current_conv = conv["messages"]
        tokenized_responses = []
        for msg in current_conv:
            if msg["role"] == "assistant":
                tokenized_responses.append(tokenizer.encode(msg["content"], add_special_tokens=False))

        tokenized_conv = tokenizer.apply_chat_template(current_conv, chat_template=conversation_template, max_length=max_tokens, truncation=True)
        all_input_ids.append(torch.LongTensor(tokenized_conv))

    return dict(input_ids=all_input_ids, labels=all_input_ids)