from dataclasses import dataclass, field
from typing import Optional

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser, LlamaForCausalLM, LlamaTokenizer


@dataclass
class ScriptArguments:
    """
    The input names representing the Adapter and Base model fine-tuned with PEFT, and the output name representing the
    merged model.
    """

    adapter_model_name: Optional[str] = field(default=None, metadata={"help": "the adapter name"})
    base_model_name: Optional[str] = field(default=None, metadata={"help": "the base model name"})
    base_tokenizer_name: Optional[str] = field(default=None, metadata={"help": "the tokeniser model name"})
    output_name: Optional[str] = field(default=None, metadata={"help": "the merged model name"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
assert script_args.adapter_model_name is not None, "please provide the name of the Adapter you would like to merge"
assert script_args.base_model_name is not None, "please provide the name of the Base model"
assert script_args.base_tokenizer_name is not None, "please provide the name of the tokenizer model"
assert script_args.output_name is not None, "please provide the output name of the merged model"

peft_config = PeftConfig.from_pretrained(script_args.adapter_model_name)
# if peft_config.task_type == "SEQ_CLS":
#     # The sequence classification task is used for the reward model in PPO
#     model = AutoModelForSequenceClassification.from_pretrained(
#         script_args.base_model_name, num_labels=1, torch_dtype=torch.bfloat16
#     )
# else:
model = LlamaForCausalLM.from_pretrained(
    script_args.base_model_name, return_dict=True, torch_dtype=torch.bfloat16
)

tokenizer = LlamaTokenizer.from_pretrained(script_args.base_model_name)
print("Loading PEFT")
# Load the PEFT model
model = PeftModel.from_pretrained(model, script_args.adapter_model_name)
model.eval()

print("Started Merging")
model = model.merge_and_unload()
print("Saving the Model")

model.save_pretrained(f"{script_args.output_name}")
tokenizer.save_pretrained(f"{script_args.output_name}")

print("Saving complete complete")
# model.push_to_hub(f"{script_args.output_name}", use_temp_dir=False)