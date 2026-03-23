import os
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def create_rotate_data_final(data_dir, checkpoint_path, top_k=20):
    print("🚀 终极修复版：已彻底消灭 Relation=0 毒瘤 Bug，准备起飞...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # 1. 加载 Checkpoint
    # ==========================================
    print(f"正在读取模型文件: {checkpoint_path} ...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    ent_embeds = state_dict['entity_embedding'].float().to(device)
    rel_embeds = state_dict['relation_embedding'].float().to(device)

    # 从你的 checkpoint 精准提取参数
    gamma = state_dict['gamma'].item() if 'gamma' in state_dict else 24.0
    embedding_range = state_dict['embedding_range'].item()
    pi = 3.14159265358979323846

    # ==========================================
    # 2. 严谨加载 实体 和 关系 字典！(Bug 修复处)
    # ==========================================
    ent2id, rel2id = {}, {}
    with open(os.path.join(data_dir, 'entities.dict'), 'r', encoding='utf-8') as f:
        for line in f:
            idx, ent = line.strip().split('\t')
            ent2id[ent] = int(idx)

    with open(os.path.join(data_dir, 'relations.dict'), 'r', encoding='utf-8') as f:
        for line in f:
            idx, rel = line.strip().split('\t')
            rel2id[rel] = int(idx)

    out_dir = os.path.join(data_dir, 'RotatE')
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'entity2id.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{len(ent2id)}\n")
        for ent, idx in ent2id.items(): f.write(f"{ent}\t{idx}\n")
    with open(os.path.join(out_dir, 'relation2id.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{len(rel2id)}\n")
        for rel, idx in rel2id.items(): f.write(f"{rel}\t{idx}\n")

    # ==========================================
    # 3. 构建 Filter 字典 (已恢复真正的 r_idx)
    # ==========================================
    print("正在构建全局 Filter 字典...")
    tail_filter = defaultdict(set)
    head_filter = defaultdict(set)
    valid_triplets, test_triplets = [], []

    for split, target_list in [('train.txt', []), ('valid.txt', valid_triplets), ('test.txt', test_triplets)]:
        file_path = os.path.join(data_dir, split)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    h_str, r_str, t_str = line.strip().split('\t')
                    # 【史诗级修复】老老实实获取真正的 r_idx！
                    h_idx, r_idx, t_idx = ent2id[h_str], rel2id[r_str], ent2id[t_str]
                    target_list.append((h_idx, r_idx, t_idx))
                    tail_filter[(h_idx, r_idx)].add(t_idx)
                    head_filter[(r_idx, t_idx)].add(h_idx)

    for out_name, target_list in [('valid2id.txt', valid_triplets), ('test2id.txt', test_triplets)]:
        with open(os.path.join(out_dir, out_name), 'w', encoding='utf-8') as f:
            f.write(f"{len(target_list)}\n")
            for h, r, t in target_list: f.write(f"{h} {t} {r}\n")

    np.save(os.path.join(out_dir, 'embeds_ent.npy'), ent_embeds.cpu().numpy())
    np.save(os.path.join(out_dir, 'embeds_rel.npy'), rel_embeds.cpu().numpy())

    # ==========================================
    # 4. 核心计算 (复刻 model.py 里的 L2 范数逻辑)
    # ==========================================
    print("🔥 开始进行精确的复数空间旋转与距离计算...")
    num_queries = len(valid_triplets) + len(test_triplets)

    all_re_ent, all_im_ent = torch.chunk(ent_embeds, 2, dim=-1)

    embed_h_fake = np.zeros((num_queries, ent_embeds.shape[-1]), dtype=np.float32)
    embed_r_fake = np.zeros((num_queries, ent_embeds.shape[-1]), dtype=np.float32)
    embed_t_fake = np.zeros((num_queries, ent_embeds.shape[-1]), dtype=np.float32)

    rank_head, rank_tail = np.zeros(num_queries, dtype=np.int32), np.zeros(num_queries, dtype=np.int32)
    topk_head, topk_tail = np.zeros((num_queries, top_k), dtype=np.int32), np.zeros((num_queries, top_k),
                                                                                    dtype=np.int32)
    topk_head_scores, topk_tail_scores = np.zeros((num_queries, top_k), dtype=np.float32), np.zeros(
        (num_queries, top_k), dtype=np.float32)

    for idx, (h_idx, r_idx, t_idx) in tqdm(enumerate(valid_triplets + test_triplets), total=num_queries):
        h_vec = ent_embeds[h_idx]
        r_vec = rel_embeds[r_idx]
        t_vec = ent_embeds[t_idx]

        re_h, im_h = torch.chunk(h_vec, 2, dim=-1)
        re_t, im_t = torch.chunk(t_vec, 2, dim=-1)

        phase_r = r_vec / (embedding_range / pi)
        re_r, im_r = torch.cos(phase_r), torch.sin(phase_r)

        # ====== 尾实体计算 (Tail Query) ======
        re_q_tail = re_h * re_r - im_h * im_r
        im_q_tail = re_h * im_r + im_h * re_r

        diff_re_tail = re_q_tail.unsqueeze(0) - all_re_ent
        diff_im_tail = im_q_tail.unsqueeze(0) - all_im_ent

        # 【同步 run.py】使用 L2 范数: score.norm(dim=0)
        norm_tail = torch.norm(torch.stack([diff_re_tail, diff_im_tail], dim=0), dim=0)
        tail_scores = gamma - norm_tail.sum(dim=1)

        filter_ts = list(tail_filter[(h_idx, r_idx)])
        if filter_ts:
            mask_ts = [x for x in filter_ts if x != t_idx]
            if mask_ts: tail_scores[mask_ts] = -float('inf')

        sorted_tail_scores, tail_sorted_indices = torch.sort(tail_scores, descending=True)
        topk_tail[idx] = tail_sorted_indices[:top_k].cpu().numpy()
        topk_tail_scores[idx] = sorted_tail_scores[:top_k].cpu().numpy()
        rank_tail[idx] = (tail_sorted_indices == t_idx).nonzero(as_tuple=True)[0].item() + 1

        # ====== 头实体计算 (Head Query) ======
        im_r_inv = -im_r
        re_q_head = re_t * re_r - im_t * im_r_inv
        im_q_head = re_t * im_r_inv + im_t * re_r

        diff_re_head = re_q_head.unsqueeze(0) - all_re_ent
        diff_im_head = im_q_head.unsqueeze(0) - all_im_ent

        # 【同步 run.py】使用 L2 范数
        norm_head = torch.norm(torch.stack([diff_re_head, diff_im_head], dim=0), dim=0)
        head_scores = gamma - norm_head.sum(dim=1)

        filter_hs = list(head_filter[(r_idx, t_idx)])
        if filter_hs:
            mask_hs = [x for x in filter_hs if x != h_idx]
            if mask_hs: head_scores[mask_hs] = -float('inf')

        sorted_head_scores, head_sorted_indices = torch.sort(head_scores, descending=True)
        topk_head[idx] = head_sorted_indices[:top_k].cpu().numpy()
        topk_head_scores[idx] = sorted_head_scores[:top_k].cpu().numpy()
        rank_head[idx] = (head_sorted_indices == h_idx).nonzero(as_tuple=True)[0].item() + 1

        # 写入木马数据
        embed_h_fake[idx] = torch.cat([re_q_tail, im_q_tail], dim=-1).cpu().numpy()
        embed_t_fake[idx] = torch.cat([re_q_head, im_q_head], dim=-1).cpu().numpy()

    print("正在保存排名数据到本地...")
    np.save(os.path.join(out_dir, 'embed_h.npy'), embed_h_fake)
    np.save(os.path.join(out_dir, 'embed_r.npy'), embed_r_fake)
    np.save(os.path.join(out_dir, 'embed_t.npy'), embed_t_fake)
    np.save(os.path.join(out_dir, 'rank_head.npy'), rank_head)
    np.save(os.path.join(out_dir, 'topk_head.npy'), topk_head)
    np.save(os.path.join(out_dir, 'topk_head_scores.npy'), topk_head_scores)
    np.save(os.path.join(out_dir, 'rank_tail.npy'), rank_tail)
    np.save(os.path.join(out_dir, 'topk_tail.npy'), topk_tail)
    np.save(os.path.join(out_dir, 'topk_tail_scores.npy'), topk_tail_scores)

    print("✅ 完美收官！请立刻去跑 data_construct_rotate.py！")


if __name__ == '__main__':
    # 替换为你实际的路径
    dataset_directory = "."
    checkpoint_file = "checkpoint"  # 请指向你真实的 checkpoint！
    create_rotate_data_final(dataset_directory, checkpoint_file)