import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
import json
import random
import torch
import numpy as np
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    SiglipVisionConfig,
    Qwen2ForCausalLM,
    SiglipVisionModel,
    SiglipImageProcessor,
)
from mini_llava import MiniLlava


# 构建dataset
class MiniDataset(Dataset):
    def __init__(self,
                 train_path,
                 tokenizer,
                 processor):
        with open(train_path, "r") as fp:
            self.data = json.load(fp)
        self.root = "/data/gongoubo/MiniLlava/data"
        self.system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        self.tokenizer = tokenizer
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        d = self.data[item]
        image = d["image"]
        texts = d["caption"]
        text = random.choice(texts)

        image = image.replace("\\", "/")
        image = os.path.join(self.root, image)
        image = Image.open(image).convert("RGB")

        image_input = self.processor(image)
        text = f"<start_of_text>{self.system} USER\n<image>\n请简单描述下图片。\nASSISTANT\n{text}<end_of_text>"
        text_input = self.tokenizer.encode(text,
                                           max_length=128,
                                           padding="max_length",
                                           add_special_tokens=False)

        out = {
            "text": torch.tensor(text_input),
            "image": torch.from_numpy(image_input["pixel_values"][0])
        }

        return out


cfg_path = "/data/gongoubo/MiniLlava/model_hub/Zhang199/TinyLLaVA-Qwen2-0.5B-SigLIP/config.json"
with open(cfg_path, "r") as fp:
    cfg = json.load(fp)

language_model_name = "Qwen/Qwen2-0.5B"
vision_model_name = "google/siglip-so400m-patch14-384"
tokenizer = AutoTokenizer.from_pretrained(language_model_name)

# print(tokenizer.encode("ASSISTANT"))
# print(tokenizer.encode("\n"))
# print(tokenizer.eos_token)
# print(tokenizer.bos_token)
# print(tokenizer.pad_token)
tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
tokenizer.add_special_tokens({"additional_special_tokens": ["ASSISTANT"]})
image_id = tokenizer.encode("<image>")[0]
processor = SiglipImageProcessor.from_pretrained(vision_model_name)

num_train_epochs = 200
train_batch_size = 2

# 构建dataloader
train_path = "data/cn_train.json"
train_dataset = MiniDataset(train_path, tokenizer, processor)

print(len(train_dataset))

minllava = MiniLlava(cfg,
                     tokenizer=tokenizer,
                     processor=processor,
                     use_pretrained=True)

minllava.language_model.resize_token_embeddings(len(tokenizer))

minllava.lock_image()

# for k, v in minllava.named_parameters():
#     print(k, v.shape)

# print(train_dataset[0])
# print(tokenizer.decode([    27,   2468,   3575,   4326,  23465,   6236,   1948,    264,  22208,
#            1196,    323,    458,  20443,  11229,  17847,     13,    576,  17847,
#            6696,  10950,     11,  11682,     11,    323,  47787,  11253,    311,
#             279,   1196,    594,   4755,     13,  13872,     25,    220, 151646,
#             198,  14880, 100405,  53481,  16872,  45930,   8997,   4939,   3846,
#            2821,     25,  73562, 100167, 102605,  99803,   9370, 100737,  89393,
#             408,   3575,   4326,     29]))

output_dir = './checkpoints/'

training_args = TrainingArguments(
    output_dir=output_dir,  # output directory 结果输出地址
    num_train_epochs=num_train_epochs,  # total # of training epochs 训练总批次
    per_device_train_batch_size=train_batch_size,  # batch size per device during training 训练批大小
    logging_dir='./logs/',  # directory for storing logs 日志存储位置
    learning_rate=3e-5,  # 学习率
    save_steps=False,  # 不保存检查点
    save_safetensors=False,
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=1,
    max_grad_norm=1,
    do_eval=False,
    do_train=True,
    report_to="none"
)


class MiniTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # text = inputs["text"]
        # image = inputs["image"]
        output = model(**inputs)
        loss = output.loss
        return loss


trainer = MiniTrainer(
    model=minllava,  # the instantiated 🤗 Transformers model to be trained 需要训练的模型
    args=training_args,  # training arguments, defined above 训练参数
    train_dataset=train_dataset,  # training dataset 训练集
)

trainer.train()
# trainer.save_pretrained(output_dir)

torch.save(trainer.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

tokenizer.save_pretrained(output_dir)
