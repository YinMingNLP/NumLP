import os
from typing import List, Optional, Dict, Sequence, Union

import torch
from torch import nn
from transformers import AutoTokenizer, PreTrainedModel, GenerationConfig
from transformers.activations import ACT2FN
from peft import PeftModel


class EmbeddingModel(nn.Module):
    def __init__(
            self,
            embedding_dir: str,
            input_size: int,
            num_feature_size: int,  # numeric feature number
            intermediate_size: int = 1024,
            output_size: int = 5120,
            hidden_act: str = 'silu',
            num_bins: int = 10,  # bin number
            bin_dim: int = 16,  # bin dimension
    ) -> None:
        super().__init__()
        entity_embedding_path = os.path.join(embedding_dir, 'entity_embeddings.pt')
        query_embedding_path = os.path.join(embedding_dir, 'query_embeddings.pt')
        numerical_embedding_path = os.path.join(embedding_dir, 'numerical_features.pt')

        self.num_feature_size = num_feature_size
        self.num_bins = num_bins
        self.bin_dim = bin_dim

        # 1. 结构特征 (KGE)
        entity_embeddings = torch.load(entity_embedding_path)
        entity_embeddings.requires_grad = False
        self.ent_embeddings = nn.Embedding.from_pretrained(entity_embeddings)

        query_embeddings = torch.load(query_embedding_path)
        query_embeddings.requires_grad = False
        self.query_embeddings = nn.Embedding.from_pretrained(query_embeddings)

        # 2. [NEW] 数值特征分箱与 Embedding 初始化
        if os.path.exists(numerical_embedding_path):
            # (num_entities, 104) 的浮点数百分位排名矩阵 [0.0 ~ 1.0]
            numerical_features = torch.load(numerical_embedding_path)
            assert numerical_features.shape[1] == num_feature_size

            # 将 [0.0, 1.0] 映射到 [0, num_bins - 1] 的整数索引
            # 例如 10个箱子: 0.15 -> 箱 1; 0.99 -> 箱 9
            bin_indices = (numerical_features * num_bins).long()
            bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)

            # 注册为 buffer，这样它会自动跟随模型移动到 GPU 上，且不参与梯度更新
            self.register_buffer('bin_indices', bin_indices)

            # 核心：每个属性的每个箱子都有独立向量，总共有 F * n 个向量
            # 例如 104个属性 * 10个箱子 = 1040 个独立词表，每个词表 32 维
            self.attr_bin_embeddings = nn.Embedding(num_feature_size * num_bins, bin_dim)
            torch.nn.init.xavier_uniform_(self.attr_bin_embeddings.weight)

            # 预计算属性偏移量: [0, n, 2n, 3n ... (F-1)n]
            # 用于让第 i 个属性去查它专属的那 n 个箱子
            feature_offsets = torch.arange(num_feature_size) * num_bins
            self.register_buffer('feature_offsets', feature_offsets)
        else:
            raise FileNotFoundError(f"Need {numerical_embedding_path} for numerical fusion!")

        # 结构翻译官
        self.struct_adapter = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=intermediate_size, bias=False),
            ACT2FN[hidden_act],
            nn.Linear(in_features=intermediate_size, out_features=output_size, bias=False),
        )

        # [NEW] 数值翻译官的输入维度变成了 F * b (例如 104 * 32 = 3328 维)
        self.num_adapter = nn.Sequential(
            nn.Linear(in_features=num_feature_size * bin_dim, out_features=intermediate_size, bias=False),
            ACT2FN[hidden_act],
            nn.Linear(in_features=intermediate_size, out_features=output_size, bias=False),
        )

        for adapter in [self.struct_adapter, self.num_adapter]:
            for layer in adapter:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, query_ids, entity_ids):
        """
        Args:
            query_ids: (batch_size, )
            entity_ids: (batch_size * K, )
        """
        # --- 获取结构 Embedding ---
        query_struct = self.query_embeddings(query_ids)  # (batch, 1000)
        ent_struct = self.ent_embeddings(entity_ids)  # (batch*K, 1000)

        # --- [NEW] 获取数值分箱 Embedding ---
        # 1. 查表获取当前 batch 中实体的 箱子索引矩阵: shape (batch*K, 104)
        ent_bin_idx = self.bin_indices[entity_ids]

        # 2. 加上偏移量，确保属性 i 查找的是属于它的 10 个箱子
        ent_bin_idx = ent_bin_idx + self.feature_offsets  # broadcasting

        # 3. Embedding 查表: shape (batch*K, 104, 32)
        ent_num_bins = self.attr_bin_embeddings(ent_bin_idx)

        # 4. 展平，把 104 个属性的向量拼接成长条: shape (batch*K, 104 * 32)
        ent_num_flat = ent_num_bins.view(ent_bin_idx.shape[0], -1)

        # --- 经过 Adapter 投影升维 ---
        query_embeds = self.struct_adapter(query_struct)  # (batch, 5120)
        ent_struct_llm = self.struct_adapter(ent_struct)  # (batch*K, 5120)
        ent_num_llm = self.num_adapter(ent_num_flat)  # (batch*K, 5120)

        # --- 残差相加对齐 ---
        entity_embeds = ent_struct_llm + ent_num_llm

        return query_embeds, entity_embeds


class KGEQwen(nn.Module):
    def __init__(
            self,
            tokenizer: AutoTokenizer,
            llm_model: Union[PreTrainedModel, PeftModel],
            kge_model: EmbeddingModel,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.llm_model = llm_model
        self.kge_model = kge_model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.LongTensor] = None,
            query_ids: Optional[torch.LongTensor] = None,
            entity_ids: Optional[torch.LongTensor] = None,
    ):
        query_holder = self.tokenizer.convert_tokens_to_ids(['[QUERY]'])[0]
        entity_holder = self.tokenizer.convert_tokens_to_ids(['[ENTITY]'])[0]

        query_position = torch.nonzero(input_ids == query_holder)  # (batch_size, 2)
        entity_position = torch.nonzero(input_ids == entity_holder)  # (batch_size*K, 2)

        query_embeds, entity_embeds = self.kge_model(query_ids, entity_ids.view(-1))

        input_ids[input_ids == query_holder] = self.tokenizer.pad_token_id
        input_ids[input_ids == entity_holder] = self.tokenizer.pad_token_id

        inputs_embeds = self.llm_model.get_input_embeddings()(input_ids).clone()

        inputs_embeds[query_position[:, 0], query_position[:, 1]] = query_embeds
        inputs_embeds[entity_position[:, 0], entity_position[:, 1]] = entity_embeds

        return self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    def save_pretrained(self, peft_model_path):
        self.llm_model.save_pretrained(peft_model_path)
        torch.save(self.kge_model.state_dict(), os.path.join(os.path.dirname(peft_model_path), 'kge.bin'))

    def generate(
            self,
            input_ids: torch.LongTensor,
            query_ids: torch.LongTensor,
            entity_ids: torch.LongTensor,
            generation_config: GenerationConfig,
            attention_mask: Optional[torch.Tensor] = None,
            pad_token_id: Optional[int] = None,
    ):
        query_holder = self.tokenizer.convert_tokens_to_ids(['[QUERY]'])[0]
        entity_holder = self.tokenizer.convert_tokens_to_ids(['[ENTITY]'])[0]

        query_position = torch.nonzero(input_ids == query_holder)
        entity_position = torch.nonzero(input_ids == entity_holder)

        query_embeds, entity_embeds = self.kge_model(query_ids, entity_ids.view(-1))

        input_ids[input_ids == query_holder] = self.tokenizer.pad_token_id
        input_ids[input_ids == entity_holder] = self.tokenizer.pad_token_id
        inputs_embeds = self.llm_model.get_input_embeddings()(input_ids).clone()

        inputs_embeds[query_position[:, 0], query_position[:, 1]] = query_embeds
        inputs_embeds[entity_position[:, 0], entity_position[:, 1]] = entity_embeds

        return self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
            generation_config=generation_config,
        )