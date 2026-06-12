# How to run the dataset creation script

`sbatch data_mod_vllm.sh openai/gpt-oss-120b structured_notes.json 256`

## Script arguments explained
- openai/gpt-oss-120b - LLM used to augment the dataset
- structured_notes.json - JSON file name (used in finetuning codes)
- 256 - batch size (how many queries sent to vLLM server at once)

# How to run the finetuning script

`sbatch finetune_new/train.sh`

All the required variables are determined in the `train.sh` script