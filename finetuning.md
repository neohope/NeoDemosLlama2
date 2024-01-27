Finetuning
============
1. 调整配置文件
```shell
# training.py
# 包括以下配置
model_name: str="PATH/to/LLAMA 2/7B"
enable_fsdp: bool= False
run_validation: bool=True
batch_size_training: int=4
gradient_accumulation_steps: int=1
num_epochs: int=3
num_workers_dataloader: int=2
lr: float=2e-4
weight_decay: float=0.0
gamma: float= 0.85
use_fp16: bool=False
mixed_precision: bool=True
val_batch_size: int=4
dataset = "samsum_dataset" # alpaca_dataset,grammar_dataset
peft_method: str = "lora" # None , llama_adapter, prefix
use_peft: bool=False
output_dir: str = "./ft-output"
freeze_layers: bool = False
num_freeze_layers: int = 1
quantization: bool = False
one_gpu: bool = False
save_model: bool = False
dist_checkpoint_root_folder: str="model_checkpoints"
dist_checkpoint_folder: str="fine-tuned"
save_optimizer: bool=False
```

2. Single GPU
```shell
# 指定GPU
# export CUDA_VISIBLE_DEVICES=GPU:id
# 单GPU调优
# --use_peft       是否启用PEFT（Parameter-Efficient Fine-Tuning）
# --peft_method    可选项为lora、llama_adapter、prefix
# --quantization   是否启用int8量化
python -m llama_recipes.finetuning --use_peft --peft_method lora --quantization --use_fp16 --model_name /patht_of_model_folder/7B --output_dir Path/to/save/PEFT/model

# 训练时可以指定数据集
--dataset grammar_dataset
--dataset alpaca_dataset
--dataset samsum_dataset
```

3. Multiple GPUs
```shell
# LORA调优，多GPU单节点
# --enable_fsdp     是否启用FSDP（Fully Sharded Data Parallel）
# --use_peft        是否启用PEFT（Parameter-Efficient Fine-Tuning）
# --peft_method     可选项为lora、llama_adapter、prefix
torchrun --nnodes 1 --nproc_per_node 4  llama/finetuning.py --enable_fsdp --use_peft --peft_method lora --model_name /patht_of_model_folder/7B --output_dir Path/to/save/PEFT/model

# LORA调优，节约内存，多GPU单节点
# --use_fast_kernels 启用Flash Attention或Xformer memory-efficient kernels
torchrun --nnodes 1 --nproc_per_node 4  llama/finetuning.py --enable_fsdp --use_peft --peft_method lora --use_fast_kernels --model_name /patht_of_model_folder/7B --output_dir Path/to/save/PEFT/model

# 全参数调优70B大模型时，仅启用FSDP，多GPU单节点
torchrun --nnodes 1 --nproc_per_node 8  llama/finetuning.py --enable_fsdp --pure_bf16 --use_fast_kernels --model_name /patht_of_model_folder/7B --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned

# 全参数调优70B大模型时，仅启用FSDP，大量节约内存，多GPU单节点
# --low_cpu_fsdp 节约cpu内存
torchrun --nnodes 1 --nproc_per_node 8 llama/finetuning.py --enable_fsdp --low_cpu_fsdp --pure_bf16 --batch_size_training 1 --model_name /patht_of_model_folder/70B --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned

# 使用slurm脚本，多GPU多个节点
# Change the num nodes and GPU per nodes in the script before running.
sbatch llama/multi_node.slurm

# 训练时可以指定数据集
--dataset grammar_dataset
--dataset alpaca_dataset
--dataset samsum_dataset
```

4. 使用自定义数据集
```shell
第一步，参考custom_dataset.py文件，完成如下签名的函数，并准备好数据集
def get_custom_dataset(dataset_config, tokenizer, split: str):

第二步，在调优时，传递custom_dataset对应参数
--dataset "custom_dataset" --custom_dataset.file "llama/custom_dataset.py"

当然文件名和函数名也可以自定义，比如文件名为neohope_dataset，函数名为get_neohope_dataset
需要首先在neohope_dataset.py文件中，完成函数
def get_neohope_dataset(dataset_config, tokenizer, split: str):
并在微调时，传递正确的参数
--dataset "custom_dataset" --custom_dataset.file "llama/neohope_dataset.py:get_neohope_dataset"
```

5. 调优的三种方式
```shell
Keep the pretrained model frozen and only fine-tune the task head for example, the classifier model.
Keep the pretrained model frozen and add a few fully connected layers on the top.
Fine-tuning on all the layers.
```
