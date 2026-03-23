import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def create_conve_bulletproof():
    print("================ 🚀 ConvE 2D卷积满血复原版 ================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ⚠️⚠️⚠️ 极其关键的 3 大路径配置 ⚠️⚠️⚠️

    # 1. 保存 conve.model 和 conve_entities.dict (JSON) 的 out 文件夹
    KGE_TRAIN_DIR = "."

    # 2. DIFT 存放原始 train.txt 的目录
    DIFT_DATA_DIR = "."

    # 3. DIFT 期望读取数据的输出目录
    DIFT_OUT_DIR = os.path.join(DIFT_DATA_DIR, "ConvE")
    os.makedirs(DIFT_OUT_DIR, exist_ok=True)

    # ==========================================
    # 1. 严格解析 JSON 格式的原配字典
    # ==========================================
    print("正在解析 JSON 格式的 ConvE 原配字典...")
    # 注意字典名字要和你 main.py 保存的对应
    with open(os.path.join(KGE_TRAIN_DIR, 'conve_entities.dict'), 'r', encoding='utf-8') as f:
        ent2id_kge = json.load(f)
    with open(os.path.join(KGE_TRAIN_DIR, 'conve_relations.dict'), 'r', encoding='utf-8') as f:
        rel2id_kge = json.load(f)

    # 翻译为 DIFT 的 TSV 格式
    with open(os.path.join(DIFT_OUT_DIR, 'entity2id.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{len(ent2id_kge)}\n")
        for ent, idx in ent2id_kge.items(): f.write(f"{ent}\t{idx}\n")
    with open(os.path.join(DIFT_OUT_DIR, 'relation2id.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{len(rel2id_kge)}\n")
        for rel, idx in rel2id_kge.items(): f.write(f"{rel}\t{idx}\n")

    # ==========================================
    # 2. 完美重建 ConvE 组件 (使用 Functional API 免去类加载)
    # ==========================================
    ckpt_path = os.path.join(KGE_TRAIN_DIR, 'conve.model')
    print(f"正在读取模型文件: {ckpt_path} ...")
    state_dict = torch.load(ckpt_path, map_location=device)

    # 提取 Embedding 和 Bias
    E_weight = state_dict['E.weight'].float()
    R_weight = state_dict['R.weight'].float()
    b_bias = state_dict['b'].float()
    d_e = E_weight.shape[1]

    # ConvE 的 2D 变形尺寸 (根据你的 model.py 默认 emb_dim1 = 20)
    emb_dim1 = 20
    emb_dim2 = d_e // emb_dim1

    # 提取卷积核与全连接层参数
    conv1_w = state_dict['conv1.weight'].float()
    conv1_b = state_dict.get('conv1.bias', None)
    if conv1_b is not None: conv1_b = conv1_b.float()
    fc_w = state_dict['fc.weight'].float()
    fc_b = state_dict['fc.bias'].float()

    # 提取所有 BatchNorm 统计量
    def get_bn(name):
        return (state_dict[f'{name}.weight'].float(), state_dict[f'{name}.bias'].float(),
                state_dict[f'{name}.running_mean'].float(), state_dict[f'{name}.running_var'].float())

    bn0_w, bn0_b, bn0_rm, bn0_rv = get_bn('bn0')
    bn1_w, bn1_b, bn1_rm, bn1_rv = get_bn('bn1')
    bn2_w, bn2_b, bn2_rm, bn2_rv = get_bn('bn2')

    # 定义前向传播核心函数 (完全跳过 Dropout，保证 eval 的绝对纯净)
    def conve_forward(h_indices, r_indices):
        e1 = E_weight[h_indices].view(-1, 1, emb_dim1, emb_dim2)
        r = R_weight[r_indices].view(-1, 1, emb_dim1, emb_dim2)

        # 1. 拼接与 BN0
        x = torch.cat([e1, r], 2)
        x = F.batch_norm(x, bn0_rm, bn0_rv, bn0_w, bn0_b, training=False)

        # 2. 卷积、BN1 与 ReLU
        x = F.conv2d(x, conv1_w, conv1_b, stride=1, padding=0)
        x = F.batch_norm(x, bn1_rm, bn1_rv, bn1_w, bn1_b, training=False)
        x = F.relu(x)

        # 3. 展平与 FC 层
        x = x.view(x.shape[0], -1)
        x = F.linear(x, fc_w, fc_b)

        # 4. BN2 与 ReLU (获得木马 Query 向量)
        x = F.batch_norm(x, bn2_rm, bn2_rv, bn2_w, bn2_b, training=False)
        q = F.relu(x)

        # 5. 打分
        scores = torch.mm(q, E_weight.transpose(1, 0)) + b_bias
        return torch.sigmoid(scores).squeeze(0), q.squeeze(0)

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
    # 4. 核心计算
    # ==========================================
    top_k = 20
    print("🔥 开始执行 ConvE 原汁原味的前向传播...")
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
            h_tensor = torch.tensor([h_idx], device=device)
            r_tensor = torch.tensor([r_idx], device=device)
            tail_scores, q_tail = conve_forward(h_tensor, r_tensor)

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

            # 把尾实体 t_idx 当作 h_indices 传入，配合逆关系
            t_tensor = torch.tensor([t_idx], device=device)
            r_inv_tensor = torch.tensor([r_inv_idx], device=device)
            head_scores, q_head = conve_forward(t_tensor, r_inv_tensor)

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
            embed_h_fake[idx] = q_tail.cpu().numpy()
            embed_t_fake[idx] = q_head.cpu().numpy()

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

    print("✅ ConvE 逆关系与卷积层解密成功！请前往 DIFT 继续处理！")


if __name__ == '__main__':
    create_conve_bulletproof()