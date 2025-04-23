import json
import re
import torch

import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    SiglipVisionModel,
    SiglipVisionConfig,
    SiglipImageProcessor,
    AutoConfig,
    PretrainedConfig,
    Qwen2ForCausalLM,
    Qwen2Config,
)
from PIL import Image


ACT_TYPE = {
    'relu': nn.ReLU,
    'gelu': nn.GELU
}

class MiniLlava(nn.Module):
    def __init__(self,
                 cfg,
                 tokenizer=None,
                 processor=None,
                 use_pretrained=False):
        super(MiniLlava, self).__init__()
        vision_cfg = cfg["vision_config"]
        language_cfg = cfg["text_config"]
        language_model_name = language_cfg["_name_or_path"]
        if use_pretrained:
            self.language_model = Qwen2ForCausalLM.from_pretrained(language_model_name, cache_dir="./model_hub")
        else:
            language_config = Qwen2Config.from_pretrained(language_model_name)
            self.language_model = Qwen2ForCausalLM._from_config(language_config)

        vision_cfg = cfg["vision_config"]
        vision_model_name = vision_cfg["model_name_or_path"]

        if use_pretrained:
            self.vision_tower = SiglipVisionModel.from_pretrained(vision_model_name, cache_dir="./model_hub")
        else:
            vision_config = SiglipVisionConfig.from_pretrained(vision_model_name)
            self.vision_tower = SiglipVisionModel(vision_config)

        connector_type = cfg["connector_type"]
        vision_hidden_size = cfg["vision_hidden_size"]
        hidden_size = cfg["hidden_size"]
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', connector_type)
        act_type = connector_type.split('_')[-1]
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(vision_hidden_size, hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(ACT_TYPE[act_type]())
            modules.append(nn.Linear(hidden_size, hidden_size))

        self.connector = nn.Sequential(*modules)

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
        else:
            self.tokenizer = tokenizer

        self.image_id = self.tokenizer.encode("<image>")[0]
        self.assistant_id = self.tokenizer.encode("ASSISTANT")[0]

        print("self.image_id: ", self.image_id)
        print("self.assistant_id: ", self.assistant_id)

        if processor is None:
            self.processor = SiglipImageProcessor.from_pretrained(vision_model_name)
        else:
            self.processor = processor

        self.used = set()

    def load_language(self, state):
        new_state = {}
        for k,v in state.items():
            if "language_model" in k:
                self.used.add(k)
                k = k.replace("language_model.", "")
                new_state[k] = v

        self.language_model.load_state_dict(new_state)

    def load_vision(self, state):
        new_state = {}
        for k,v in state.items():
            if "vision_tower" in k:
                self.used.add(k)
                k = k.replace("vision_tower._vision_tower.", "")
                new_state[k] = v

        self.vision_tower.load_state_dict(new_state)

    def load_connector(self, state):
        new_state = {}
        for k, v in state.items():
            if "connector" in k:
                self.used.add(k)
                k = k.replace("connector._connector.", "")
                new_state[k] = v

        self.connector.load_state_dict(new_state)

    def get_text_embedding(self, input_ids):
        embed_layer = self.language_model.get_input_embeddings()
        input_embeddings = embed_layer(input_ids)
        return input_embeddings

    def encode_text(self,
                    text,
                    process_text=True):
        if process_text:
            system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
            text = f"<start_of_text>{system} USER: <image>\n{text} ASSISTANT: "
            text = self.tokenizer.encode(text, max_length=1024)
            image_ind = text.index(self.image_id)
            text_embeddings = self.get_text_embedding(torch.tensor(text).unsqueeze(0))
        else:
            if not isinstance(text, torch.Tensor):
                text = self.tokenizer.encode(text, max_length=1024)
                image_ind = text.index(self.image_id)
                text_embeddings = self.get_text_embedding(torch.tensor(text).unsqueeze(0))
            else:
                image_ind = (text == self.image_id).nonzero(as_tuple=True)[1].data[0]
                text_embeddings = self.get_text_embedding(text)
        return text_embeddings, image_ind

    def encode_image(self, image_path, process_image=True):
        if process_image:
            image = Image.open(image_path).convert("RGB")
            image = self.processor(image, return_tensors="pt")
            image_embeddings = self.vision_tower(**image, output_hidden_states=True)
        else:
            inp = {"pixel_values": image_path}
            image_embeddings = self.vision_tower(**inp, output_hidden_states=True)
        return image_embeddings.hidden_states[-2]

    def projector_forward(self, image_embeddings):
        output = self.connector(image_embeddings)
        return output

    def lock_image(self):
        for param in self.vision_tower.parameters():
            param.requires_grad = False
        self.vision_tower.eval()  # 设置为评估模式
        print("Vision tower parameters frozen.")

    def forward(self, text, image):
        b, seq_len = text.shape
        all_embeddings = []
        labels = []
        for i in range(b):
            t_text = text[i, :].unsqueeze(0)
            t_image = image[i, ...].unsqueeze(0)
            text_embedding, image_ind = self.encode_text(t_text, process_text=False)
            assistant_ind = (t_text == self.assistant_id).nonzero(as_tuple=True)[1].data[0]
            t_labels = t_text[:, assistant_ind+2:]  # ASSISTANT后面还有一个\n
            # assistant_ind主要是用于构建labels
            image_embedding = self.encode_image(t_image, process_image=False)
            projector_output = self.projector_forward(image_embedding)
            _, num_image_token, _ = projector_output.shape
            # 计算assistant之前有多少个token
            # assistant_ind是索引，+1表示到该位置有多少个token，再+1表示ASSISTANT后面的\n，
            # -1表示去除掉<image>，再加上num_image_token
            pre_assistant = assistant_ind+1+1-1+num_image_token
            t_labels = [-100] * pre_assistant + t_labels.squeeze(0).detach().cpu().numpy().tolist()
            input_embeddings = []
            input_embeddings.append(text_embedding[:, :image_ind, :])
            input_embeddings.append(projector_output)
            input_embeddings.append(text_embedding[:, image_ind + 1:, :])

            input_embeddings = torch.cat(input_embeddings, 1)

            t_labels = t_labels + [self.tokenizer.pad_token_id] * (input_embeddings.shape[1] - len(t_labels))
            t_labels = torch.tensor(t_labels).unsqueeze(0)
            assert input_embeddings.shape[1] == t_labels.shape[-1]

            all_embeddings.append(input_embeddings)
            labels.append(t_labels)

        all_embeddings = torch.cat(all_embeddings, 0).to(self.language_model.device)
        labels = torch.cat(labels, 0).to(self.language_model.device)
        output = self.language_model(inputs_embeds=all_embeddings, labels=labels)
        return output

    def generate(self, text, image):
        text_embeddings, image_ind = self.encode_text(text)
        image_embeddings = self.encode_image(image)
        projector_output = self.projector_forward(image_embeddings)
        input_embeddings = []
        input_embeddings.append(text_embeddings[:, :image_ind, :])
        input_embeddings.append(projector_output)
        input_embeddings.append(text_embeddings[:, image_ind+1:, :])

        input_embeddings = torch.cat(input_embeddings, 1)

        with torch.no_grad():
            output = self.language_model.generate(inputs_embeds=input_embeddings,
                                                  max_new_tokens=512)
        output = self.tokenizer.decode(output.detach().cpu().numpy()[0])
        return output

    def generate2(self, text, image):
        text_embeddings, image_ind = self.encode_text(text, process_text=False)
        image_embeddings = self.encode_image(image)
        projector_output = self.projector_forward(image_embeddings)
        input_embeddings = []
        input_embeddings.append(text_embeddings[:, :image_ind, :])
        input_embeddings.append(projector_output)
        input_embeddings.append(text_embeddings[:, image_ind+1:, :])

        input_embeddings = torch.cat(input_embeddings, 1)
        input_embeddings = input_embeddings.to(self.language_model.dtype)
        with torch.no_grad():
            output = self.language_model.generate(inputs_embeds=input_embeddings,
                                                  max_new_tokens=512)
        output = self.tokenizer.decode(output.detach().cpu().numpy()[0])
        return output


if __name__ == '__main__':

    predict = "trained"
    if predict == "ori":
        with open("/data/gongoubo/MiniLlava/model_hub/Zhang199/TinyLLaVA-Qwen2-0.5B-SigLIP/config.json", "r") as fp:
            cfg = json.loads(fp.read())

        mini_llava = MiniLlava(cfg)

        from safetensors.torch import load_file

        ckpt = load_file("/data/gongoubo/MiniLlava/model_hub/Zhang199/TinyLLaVA-Qwen2-0.5B-SigLIP/model.safetensors")

        mini_llava.load_language(ckpt)
        mini_llava.load_vision(ckpt)
        mini_llava.load_connector(ckpt)

        # is_used = mini_llava.used
        # for k,v in ckpt.items():
        #     if k not in  is_used:
        #         print(k)

        text = "请简单描述上述图片"
        image = "data/Llama3_Repo.jpeg"
        # embeddings, ids = mini_llava.encode_text(text)
        # print(embeddings.shape, ids)
        # image_embeddings = mini_llava.encode_image(image)
        # connector_output = mini_llava.projector_forward(image_embeddings)

        mini_llava.generate(text, image)

    elif predict == "trained":
        with open("/data/gongoubo/MiniLlava/model_hub/Zhang199/TinyLLaVA-Qwen2-0.5B-SigLIP/config.json", "r") as fp:
            cfg = json.loads(fp.read())

        language_model_name = "Qwen/Qwen2-0.5B"
        tokenizer = AutoTokenizer.from_pretrained(language_model_name)

        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
        tokenizer.add_special_tokens({"additional_special_tokens": ["ASSISTANT"]})

        mini_llava = MiniLlava(cfg, tokenizer=tokenizer, use_pretrained=False)
        mini_llava.language_model.resize_token_embeddings(len(tokenizer))
        state_dict = torch.load("/data/gongoubo/MiniLlava/checkpoints/checkpoint-98500/pytorch_model.bin", map_location="cpu")
        mini_llava.load_state_dict(state_dict)

        mini_llava.eval()
        image = "data/flickr8k-images/2513260012_03d33305cf.jpg"
        # image = "data/flickr8k-images/218342358_1755a9cce1.jpg"
        # image = "data/flickr8k-images/1347519824_e402241e4f.jpg"
        """
        "caption": [
          "在雪地中奔跑的狗。",
          "正在跑的黑狗和白狗。",
          "在雪地上跑的狗。",
          "雪地上黑狗追白狗。",
          "雪地里狗。"
        ]
        """
        # embeddings, ids = mini_llava.encode_text(text)
        # print(embeddings.shape, ids)
        # image_embeddings = mini_llava.encode_image(image)
        # connector_output = mini_llava.projector_forward(image_embeddings)

        system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        text = f"<start_of_text>{system} USER\n<image>\n请简单描述下图片。\nASSISTANT\n"
        print(mini_llava.generate2(text, image))

