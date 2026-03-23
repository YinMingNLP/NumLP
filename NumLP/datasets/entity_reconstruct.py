import os
import json
from collections import defaultdict


def enhance_desc_with_sequential_numerics(entity_json_path, numerical_txt_path, output_json_path):
    """
    按照 numerical_literals.txt 中的原本顺序读取数值属性，清理冗余前缀后，
    将其追加到原有实体描述 (desc) 的末尾。保持原有 entity.json 的实体顺序。
    """
    print(f"正在加载实体文件: {entity_json_path} ...")
    with open(entity_json_path, 'r', encoding='utf-8') as f:
        entities = json.load(f)

    print(f"正在解析数值属性文件: {numerical_txt_path} ...")
    # 使用 list 来天然保持 txt 文件中读取到的顺序
    entity_numerics = defaultdict(list)

    with open(numerical_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                ent_id = parts[0]
                rel_id = parts[1]
                val = parts[2]

                # 清洗冗余属性名: 只保留最后一个 '/' 之后的内容
                # clean_rel_name = rel_id.split('/')[-1]
                # 只保留最后一个'.'之后的内容
                clean_rel_name = rel_id.split('.')[-1].replace('_', ' ')

                # 按照读取的顺序加入列表
                entity_numerics[ent_id].append(f"{clean_rel_name}: {val}")

    print("正在将数值属性追加到实体描述后面...")
    match_count = 0

    # 遍历原 entities 字典，保证实体顺序不乱
    for ent_id, ent_data in entities.items():
        if ent_id in entity_numerics:
            # 提取该实体的数值属性列表 (保留了文件中的原生顺序)
            num_list = entity_numerics[ent_id]

            # 组合成规范的字符串，例如: "[Numerical: area: 48671.0, population_number: 1000000]"
            # attr_str = "[Numerical: " + ", ".join(num_list) + "]"
            # attr_str = "Related Numerical Attributes: " + ", ".join(num_list) + "."
            attr_str = "" + ", ".join(num_list) + "."

            # 将数值属性【追加】到 desc 的后面
            if ent_data.get('desc'):
                # 如果原本有描述，在原本描述后加一个空格再拼接
                # 后置数值属性
                # entities[ent_id]['desc'] = ent_data['desc'] + " " + attr_str
                # 前置数值属性
                entities[ent_id]['desc'] = attr_str + " " + ent_data['desc']
            else:
                # 如果原本没有描述，直接赋值
                entities[ent_id]['desc'] = attr_str

            match_count += 1

    print(f"正在保存更新后的文件: {output_json_path} ...")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(entities, f, indent=4, ensure_ascii=False)

    print(f"处理完成！共有 {match_count} 个实体成功在描述末尾追加了数值属性。")


if __name__ == '__main__':
    # --- 路径设置 ---
    dataset_dir = "FB15K237"  # 替换为你的数据集目录

    entity_json_path = os.path.join(dataset_dir, "entity.json")
    numerical_txt_path = os.path.join(dataset_dir, "numerical_literals.txt")
    output_json_path = os.path.join(dataset_dir, "entity_enhanced.json")

    if os.path.exists(entity_json_path) and os.path.exists(numerical_txt_path):
        enhance_desc_with_sequential_numerics(entity_json_path, numerical_txt_path, output_json_path)
    else:
        print("错误: 找不到 entity.json 或 numerical_literals.txt，请检查路径。")