from transformers import Trainer
import torch
import os




class MambaTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.pop("input_ids")
        lm_logits = model(input_ids).logits

        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = input_ids.to(lm_logits.device)
            labels = labels[:, 1:]
        if "attention_mask" in inputs:
            labels = torch.where(inputs["attention_mask"], labels, -100)
        labels = labels.contiguous()
        shift_logits = lm_logits[:, :-1, :].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))


        return lm_loss


    def prediction_step(
            self,
            model,
            inputs,
            prediction_loss_only: bool,
            ignore_keys=None,
    ):
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """


        with torch.no_grad():
            loss = self.compute_loss(model=self.model, inputs=inputs)

        return (loss, None, None)


    def save_model(self, output_dir, _internal_call):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        torch.save(self.model.state_dict(), f"{output_dir}/pytorch_model.bin")
        self.model.config.to_json_file(f"{output_dir}/config.json")
        print(f"Saved model to {output_dir}/pytorch_model")
        self.tokenizer.save_pretrained(output_dir)

