import os
import json
import random


def load_samples(json_path):
    """
    从指定路径加载 JSON 文件，返回解析后的 Python 列表对象。
    假设文件最外层是一个列表，列表中的每个元素都是一个 dict（样本）。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"输入文件 {json_path} 顶层结构不是列表，请检查格式。")
    return data


def split_by_label(samples):
    """
    将样本列表按照 label 字段分为两个列表。假设每个样本都有整型字段 "label"，值为 0 或 1。
    返回一个二元组 (label0_list, label1_list)。
    """
    label0_list = []
    label1_list = []
    for item in samples:
        lbl = item.get("label")
        if lbl == 0:
            label0_list.append(item)
        elif lbl == 1:
            label1_list.append(item)
        else:
            # 如果有其他取值，忽略或抛出异常，根据需求自行调整
            continue
    return label0_list, label1_list


def sample_k_from_list(lst, k, seed=None):
    """
    从 lst 中随机抽取 k 个元素，返回一个新的列表。
    如果 len(lst) < k，则会抛出 ValueError。
    """
    if len(lst) < k:
        raise ValueError(f"待抽样的列表长度 ({len(lst)}) 小于请求数量 {k}，无法抽样。")
    if seed is not None:
        random.seed(seed)
    return random.sample(lst, k)


def write_json_list(data_list, out_path):
    """
    将 Python 列表 data_list 序列化为 JSON，并写入到 out_path。
    """
    with open(out_path, "w", encoding="utf-8") as f:
        # indent=2 可读性更好
        json.dump(data_list, f, ensure_ascii=False, indent=2)


def main():
    input_json = "***"
    output_dir = "***"

    # 1. 读取所有样本
    all_samples = load_samples(input_json)
    print(f"共加载 {len(all_samples)} 个样本。")

    # 2. 按 label 拆分
    label0_list, label1_list = split_by_label(all_samples)
    print(f"label=0 的样本数：{len(label0_list)}")
    print(f"label=1 的样本数：{len(label1_list)}")

    # 3. 分别随机抽取 150 个
    k = 150
    # 为保证两次抽样一致，可以在外层统一使用同一个 seed，再传进去
    seed0 = None
    seed1 = None
    # 如果传入一个 seed，可以在选取 label0 时用 seed，选取 label1 时用 seed+1（或其他方式）以避免序列完全相同
    if seed0 is not None:
        seed1 = seed0 + 1

    sampled_label0 = sample_k_from_list(label0_list, k, seed0)
    sampled_label1 = sample_k_from_list(label1_list, k, seed1)

    # 4. 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 5. 将抽样结果分别写入 JSON 文件
    out_file = os.path.join(output_dir, "selected_for_cluster_300.json")

    write_json_list(sampled_label0+sampled_label1, out_file)
    print(f"已将 {k*2} 个的样本写入：{out_file}")

    print("抽样并输出完成。")


if __name__ == "__main__":
    main()
