import os
import json
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def create_distmult_bulletproof():
    print("================ 🚀 DistMult 极简逐元素相乘版 ================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ⚠️⚠️⚠️ 极其关键的 3 大路径配置 ⚠️⚠️⚠️

    # 1. 保存 distmult.model 和 distmult_entities.dict (JSON) 的 out 文件夹
    KGE_TRAIN_DIR = "."

    # 2. DIFT 存放原始 train.txt 的目录
    DIFT_DATA_DIR = "."

    # 3. DIFT 期望读取数据的输出目录
    DIFT_OUT_DIR = os.path.join(DIFT_DATA_DIR, "DistMult")
    os.makedirs(DIFT_OUT_DIR, exist_ok=True)

    # ==========================================
    # 1. 严格解析 JSON 格式的原配字典
    # ==========================================
    print("正在解析 JSON 格式的 DistMult 原配字典...")
    with open(os.path.join(KGE_TRAIN_DIR, 'distmult_entities.dict'), 'r', encoding='utf-8') as f:
        ent2id_kge = json.load(f)
    with open(os.path.join(KGE_TRAIN_DIR, 'distmult_relations.dict'), 'r', encoding='utf-8') as f:
        rel2id_kge = json.load(f)

    # 翻译为 DIFT 喜欢的 TSV 格式
    with open(os.path.join(DIFT_OUT_DIR, 'entity2id.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{len(ent2id_kge)}\n")
        for ent, idx in ent2id_kge.items(): f.write(f"{ent}\t{idx}\n")
    with open(os.path.join(DIFT_OUT_DIR, 'relation2id.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{len(rel2id_kge)}\n")
        for rel, idx in rel2id_kge.items(): f.write(f"{rel}\t{idx}\n")

    # ==========================================
    # 2. 提取模型参数 (DistMult 最为纯粹，只需 E 和 R)
    # ==========================================
    ckpt_path = os.path.join(KGE_TRAIN_DIR, 'distmult.model')
    print(f"正在读取模型文件: {ckpt_path} ...")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    E_weight = state_dict['E.weight'].float().to(device)
    R_weight = state_dict['R.weight'].float().to(device)
    d_e = E_weight.shape[1]

    # ==========================================
    # 3. 构建 Filter 字典 (保证顺序严格对齐 DIFT)
    # ==========================================
    print("正在对齐考试题目并构建 Filter...")
    tail_filter = defaultdict(set)
    head_filter = defaultdict(set)
    valid_triplets, test_triplets = [], []

    for split, target_list in [('train.txt', []), ('valid.txt', valid_triplets), ('test.txt', test_triplets)]:
        file_path = os.path.join(DIFT_DATA_DIR, split)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    h_str, r_str, t_str = line.strip().split('\t')
                    h_idx, r_idx, t_idx = ent2id_kge[h_str], rel2id_kge[r_str], ent2id_kge[t_str]
                    target_list.append((h_idx, r_idx, t_idx))
                    tail_filter[(h_idx, r_idx)].add(t_idx)
                    head_filter[(r_idx, t_idx)].add(h_idx)

    for out_name, target_list in [('valid2id.txt', valid_triplets), ('test2id.txt', test_triplets)]:
        with open(os.path.join(DIFT_OUT_DIR, out_name), 'w', encoding='utf-8') as f:
            f.write(f"{len(target_list)}\n")
            for h, r, t in target_list: f.write(f"{h} {t} {r}\n")

    np.save(os.path.join(DIFT_OUT_DIR, 'embeds_ent.npy'), E_weight.cpu().numpy())
    np.save(os.path.join(DIFT_OUT_DIR, 'embeds_rel.npy'), R_weight.cpu().numpy())

    # ==========================================
    # 4. 核心计算 (极其干净的 h * r)
    # ==========================================
    top_k = 20
    print("🔥 开始执行 DistMult 逐元素相乘的前向传播...")
    num_queries = len(valid_triplets) + len(test_triplets)

    embed_h_fake = np.zeros((num_queries, d_e), dtype=np.float32)
    embed_r_fake = np.zeros((num_queries, d_e), dtype=np.float32)  # 全0木马
    embed_t_fake = np.zeros((num_queries, d_e), dtype=np.float32)

    rank_head, rank_tail = np.zeros(num_queries, dtype=np.int32), np.zeros(num_queries, dtype=np.int32)
    topk_head, topk_tail = np.zeros((num_queries, top_k), dtype=np.int32), np.zeros((num_queries, top_k),
                                                                                    dtype=np.int32)
    topk_head_scores, topk_tail_scores = np.zeros((num_queries, top_k), dtype=np.float32), np.zeros(
        (num_queries, top_k), dtype=np.float32)

    # 建立反向关系速查表
    rel_names = {v: k for k, v in rel2id_kge.items()}

    with torch.no_grad():
        for idx, (h_idx, r_idx, t_idx) in tqdm(enumerate(valid_triplets + test_triplets), total=num_queries):

            # ====== 尾实体预测 (Tail Query) ======
            h_vec = E_weight[h_idx].unsqueeze(0)  # (1, d_e)
            r_vec = R_weight[r_idx].unsqueeze(0)  # (1, d_e)

            # DistMult 核心：逐元素相乘
            q_tail = h_vec * r_vec

            # 内积打分
            raw_scores_tail = torch.mm(q_tail, E_weight.transpose(1, 0))
            tail_scores = torch.sigmoid(raw_scores_tail).squeeze(0)

            # Filter
            filter_ts = list(tail_filter[(h_idx, r_idx)])
            if filter_ts:
                mask_ts = [x for x in filter_ts if x != t_idx]
                if mask_ts: tail_scores[mask_ts] = -float('inf')

            sorted_tail_scores, tail_sorted_indices = torch.sort(tail_scores, descending=True)
            topk_tail[idx] = tail_sorted_indices[:top_k].cpu().numpy()
            topk_tail_scores[idx] = sorted_tail_scores[:top_k].cpu().numpy()
            rank_tail[idx] = (tail_sorted_indices == t_idx).nonzero(as_tuple=True)[0].item() + 1

            # ====== 头实体预测 (Head Query: 依赖逆关系) ======
            r_name = rel_names[r_idx]
            r_inv_name = r_name + "_reverse"
            if r_inv_name not in rel2id_kge:
                raise ValueError(f"找不到 {r_name} 的逆关系 '{r_inv_name}'！")
            r_inv_idx = rel2id_kge[r_inv_name]

            t_vec = E_weight[t_idx].unsqueeze(0)
            r_inv_vec = R_weight[r_inv_idx].unsqueeze(0)

            # 使用目标实体和逆关系进行相乘
            q_head = t_vec * r_inv_vec

            raw_scores_head = torch.mm(q_head, E_weight.transpose(1, 0))
            head_scores = torch.sigmoid(raw_scores_head).squeeze(0)

            # Filter
            filter_hs = list(head_filter[(r_idx, t_idx)])
            if filter_hs:
                mask_hs = [x for x in filter_hs if x != h_idx]
                if mask_hs: head_scores[mask_hs] = -float('inf')

            sorted_head_scores, head_sorted_indices = torch.sort(head_scores, descending=True)
            topk_head[idx] = head_sorted_indices[:top_k].cpu().numpy()
            topk_head_scores[idx] = sorted_head_scores[:top_k].cpu().numpy()
            rank_head[idx] = (head_sorted_indices == h_idx).nonzero(as_tuple=True)[0].item() + 1

            # 木马写入 (写入最终的 Query 向量)
            embed_h_fake[idx] = q_tail.squeeze(0).cpu().numpy()
            embed_t_fake[idx] = q_head.squeeze(0).cpu().numpy()

    print("正在保存数据...")
    np.save(os.path.join(DIFT_OUT_DIR, 'embed_h.npy'), embed_h_fake)
    np.save(os.path.join(DIFT_OUT_DIR, 'embed_r.npy'), embed_r_fake)  # 全 0
    np.save(os.path.join(DIFT_OUT_DIR, 'embed_t.npy'), embed_t_fake)
    np.save(os.path.join(DIFT_OUT_DIR, 'rank_head.npy'), rank_head)
    np.save(os.path.join(DIFT_OUT_DIR, 'topk_head.npy'), topk_head)
    np.save(os.path.join(DIFT_OUT_DIR, 'topk_head_scores.npy'), topk_head_scores)
    np.save(os.path.join(DIFT_OUT_DIR, 'rank_tail.npy'), rank_tail)
    np.save(os.path.join(DIFT_OUT_DIR, 'topk_tail.npy'), topk_tail)
    np.save(os.path.join(DIFT_OUT_DIR, 'topk_tail_scores.npy'), topk_tail_scores)

    print("✅ DistMult 完美解密！代码最少，跑得最快！")


if __name__ == '__main__':
    create_distmult_bulletproof()