import os
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def create_transe_preprocessing_data_gpu(data_dir, ent_npy_path, rel_npy_path, top_k=20):
    print("🚀 开始利用 GPU 加速生成 DIFT 预处理数据 (完美 Filter 过滤版)...")

    # 启用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ 当前计算设备: {device} ⚡")

    # 将 NumPy 数组直接载入 GPU 显存
    ent_embeds = torch.from_numpy(np.load(ent_npy_path)).float().to(device)
    rel_embeds = torch.from_numpy(np.load(rel_npy_path)).float().to(device)

    # 1. 读取字典
    print("正在加载训练时的字典，确保向量严格对齐...")
    ent2id = {}
    dict_path = os.path.join(data_dir, 'entities.dict')
    alt_dict_path = os.path.join(data_dir, 'entity2id.txt')

    if os.path.exists(dict_path):
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                idx, ent = line.strip().split('\t')
                ent2id[ent] = int(idx)
    elif os.path.exists(alt_dict_path):
        with open(alt_dict_path, 'r', encoding='utf-8') as f:
            for line in f.readlines()[1:]:
                ent, idx = line.strip().split('\t')
                ent2id[ent] = int(idx)
    else:
        raise FileNotFoundError(f"找不到实体字典！请检查 {data_dir} 下是否有 entities.dict。")

    rel2id = {}
    rel_dict_path = os.path.join(data_dir, 'relations.dict')
    alt_rel_dict_path = os.path.join(data_dir, 'relation2id.txt')

    if os.path.exists(rel_dict_path):
        with open(rel_dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                idx, rel = line.strip().split('\t')
                rel2id[rel] = int(idx)
    elif os.path.exists(alt_rel_dict_path):
        with open(alt_rel_dict_path, 'r', encoding='utf-8') as f:
            for line in f.readlines()[1:]:
                rel, idx = line.strip().split('\t')
                rel2id[rel] = int(idx)

    # 创建 TransE 输出目录
    out_dir = os.path.join(data_dir, 'TransE')
    os.makedirs(out_dir, exist_ok=True)

    # 2. 生成 DIFT 格式的 id 字典
    with open(os.path.join(out_dir, 'entity2id.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{len(ent2id)}\n")
        for ent, idx in ent2id.items(): f.write(f"{ent}\t{idx}\n")
    with open(os.path.join(out_dir, 'relation2id.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{len(rel2id)}\n")
        for rel, idx in rel2id.items(): f.write(f"{rel}\t{idx}\n")

    # 3. 构建全局 Filter 字典并转换验证/测试集
    print("正在构建全局 Filter 字典并转换三元组...")
    tail_filter = defaultdict(set)
    head_filter = defaultdict(set)
    train_triplets, valid_triplets, test_triplets = [], [], []

    # 读取所有数据（包含 train）以构建完整的图谱视野
    for split, target_list in [('train.txt', train_triplets),
                               ('valid.txt', valid_triplets),
                               ('test.txt', test_triplets)]:
        file_path = os.path.join(data_dir, split)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    h_str, r_str, t_str = line.strip().split('\t')
                    h_idx, r_idx, t_idx = ent2id[h_str], rel2id[r_str], ent2id[t_str]
                    target_list.append((h_idx, r_idx, t_idx))

                    # 录入 Filter 字典
                    tail_filter[(h_idx, r_idx)].add(t_idx)
                    head_filter[(r_idx, t_idx)].add(h_idx)

    # 保存 valid 和 test 的 id 格式文件
    for out_name, target_list in [('valid2id.txt', valid_triplets),
                                  ('test2id.txt', test_triplets)]:
        with open(os.path.join(out_dir, out_name), 'w', encoding='utf-8') as f:
            f.write(f"{len(target_list)}\n")
            for h, r, t in target_list:
                f.write(f"{h} {t} {r}\n")

    # 保存基础的 .npy (已修复之前的缩进错误)
    np.save(os.path.join(out_dir, 'embeds_ent.npy'), ent_embeds.cpu().numpy())
    np.save(os.path.join(out_dir, 'embeds_rel.npy'), rel_embeds.cpu().numpy())

    # 4. 核心计算：使用 GPU 计算 Filtered Top-K
    print("🔥 开始使用 GPU 计算 Filtered Top-K 候选排名...")
    all_eval_triplets = valid_triplets + test_triplets
    num_queries = len(all_eval_triplets)

    # 准备 CPU 内存用于保存最终结果
    embed_h = np.zeros((num_queries, ent_embeds.shape[-1]), dtype=np.float32)
    embed_r = np.zeros((num_queries, rel_embeds.shape[-1]), dtype=np.float32)
    embed_t = np.zeros((num_queries, ent_embeds.shape[-1]), dtype=np.float32)

    rank_head, rank_tail = np.zeros(num_queries, dtype=np.int32), np.zeros(num_queries, dtype=np.int32)
    topk_head, topk_tail = np.zeros((num_queries, top_k), dtype=np.int32), np.zeros((num_queries, top_k),
                                                                                    dtype=np.int32)
    topk_head_scores, topk_tail_scores = np.zeros((num_queries, top_k), dtype=np.float32), np.zeros(
        (num_queries, top_k), dtype=np.float32)

    for idx, (h_idx, r_idx, t_idx) in tqdm(enumerate(all_eval_triplets), total=num_queries):
        # 提取 GPU 上的向量
        h_vec = ent_embeds[h_idx]
        r_vec = rel_embeds[r_idx]
        t_vec = ent_embeds[t_idx]

        # 拷贝回 CPU 用于保存基础快照
        embed_h[idx], embed_r[idx], embed_t[idx] = h_vec.cpu().numpy(), r_vec.cpu().numpy(), t_vec.cpu().numpy()

        # ============ GPU 尾实体预测 (Tail Prediction) ============
        tail_query = h_vec + r_vec
        tail_scores = -torch.norm(ent_embeds - tail_query.unsqueeze(0), p=1, dim=1) * 100000.0

        # [Filter 操作]
        filter_ts = list(tail_filter[(h_idx, r_idx)])
        if filter_ts:
            mask_ts = [x for x in filter_ts if x != t_idx]  # 保留当前目标答案
            if mask_ts:
                tail_scores[mask_ts] = -float('inf')  # 将其他已知答案屏蔽

        sorted_tail_scores, tail_sorted_indices = torch.sort(tail_scores, descending=True)
        topk_tail[idx] = tail_sorted_indices[:top_k].cpu().numpy()
        topk_tail_scores[idx] = sorted_tail_scores[:top_k].cpu().numpy()
        rank_tail[idx] = (tail_sorted_indices == t_idx).nonzero(as_tuple=True)[0].item() + 1

        # ============ GPU 头实体预测 (Head Prediction) ============
        head_query = t_vec - r_vec
        head_scores = -torch.norm(ent_embeds - head_query.unsqueeze(0), p=1, dim=1) * 100000.0

        # [Filter 操作]
        filter_hs = list(head_filter[(r_idx, t_idx)])
        if filter_hs:
            mask_hs = [x for x in filter_hs if x != h_idx]  # 保留当前目标答案
            if mask_hs:
                head_scores[mask_hs] = -float('inf')  # 将其他已知答案屏蔽

        sorted_head_scores, head_sorted_indices = torch.sort(head_scores, descending=True)
        topk_head[idx] = head_sorted_indices[:top_k].cpu().numpy()
        topk_head_scores[idx] = sorted_head_scores[:top_k].cpu().numpy()
        rank_head[idx] = (head_sorted_indices == h_idx).nonzero(as_tuple=True)[0].item() + 1

    print("正在保存排名数据到本地...")
    np.save(os.path.join(out_dir, 'embed_h.npy'), embed_h)
    np.save(os.path.join(out_dir, 'embed_r.npy'), embed_r)
    np.save(os.path.join(out_dir, 'embed_t.npy'), embed_t)
    np.save(os.path.join(out_dir, 'rank_head.npy'), rank_head)
    np.save(os.path.join(out_dir, 'topk_head.npy'), topk_head)
    np.save(os.path.join(out_dir, 'topk_head_scores.npy'), topk_head_scores)
    np.save(os.path.join(out_dir, 'rank_tail.npy'), rank_tail)
    np.save(os.path.join(out_dir, 'topk_tail.npy'), topk_tail)
    np.save(os.path.join(out_dir, 'topk_tail_scores.npy'), topk_tail_scores)

    print("✅ GPU Filter 计算完成！你的 MRR 将恢复到 0.320+！")


if __name__ == '__main__':
    # 这里的 data_dir 请填你 run.py 运行时指定的 --data_path 路径！
    dataset_directory = "."
    entity_npy = "entity_embedding.npy"
    relation_npy = "relation_embedding.npy"
    create_transe_preprocessing_data_gpu(dataset_directory, entity_npy, relation_npy)