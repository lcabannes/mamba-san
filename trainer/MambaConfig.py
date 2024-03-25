from dataclasses import dataclass, asdict
from mamba_ssm.models.config_mamba import MambaConfig
import json


@dataclass
class MambaConfig(MambaConfig):
    def __init__(self, config: dict=None):
        if config is not None:
            self.d_model = config["d_model"]
            self.n_layer = config["n_layer"]
            self.vocab_size = config["vocab_size"]
            self.ssm_cfg = config["ssm_cfg"]
            self.rms_norm = config["rms_norm"]
            self.residual_in_fp32 = config["residual_in_fp32"]
            self.fused_add_norm = config["fused_add_norm"]
            self.pad_vocab_size_multiple = config["pad_vocab_size_multiple"]


    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.

        Returns:
            Dictionary of all the attributes that make up this configuration instance.
        """
        return asdict(self)

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


    def to_json_file(self, json_file_path):
        with open(json_file_path, 'w', encoding="utf-8") as writer:
            writer.write(self.to_json_string())
