import os
import json
import re
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

import torch
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel

# ------------------------------------------------------------------------------
# 1. 特征提取和向量化部分
# ------------------------------------------------------------------------------


class CodeVectorizer:
    """
    使用 HuggingFace Transformers 将代码字符串映射到语义向量 V。
    这里示例使用 'microsoft/codebert-base'，你也可以换成自己想要的预训练模型。
    """

    def __init__(self, model_name: str = "microsoft/codebert-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()

    def encode(self, code_str: str) -> np.ndarray:
        """
        对单个代码字符串进行编码，返回一个固定维度的语义向量（numpy 数组）。
        这里示例做 mean-pooling：对最后一层隐藏态在 token 维度上做平均。
        """
        inputs = self.tokenizer(
            code_str, return_tensors="pt", truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            # (batch_size=1, seq_len, hidden_size)
            last_hidden = outputs.last_hidden_state
            # mean pooling
            vec = last_hidden.mean(dim=1).squeeze(0)  # (hidden_size,)
        return vec.cpu().numpy()


def extract_features_from_node_line_sym(node_line_sym: List[str]) -> int:
    """
    从 node_line_sym 中计算 var_num：找出所有形如 VAR<number> 的标识符，
    取其中最大的数字即可作为 var_num。如果没有找到，则 var_num=0。
    """
    max_var = 0
    for line in node_line_sym:
        # 在行里查找所有 VAR 后面跟数字的模式
        for match in re.findall(r"\bVAR(\d+)\b", line):
            num = int(match)
            if num > max_var:
                max_var = num
    return max_var


def count_function_calls(node_line_sym: List[str]) -> int:
    """
    从 node_line_sym 中计算 fun_num：找出所有形如 FUN<number> 的标识符，
    取其中最大的数字即可作为 fun_num。如果没有找到，则 fun_num=0。
    """
    max_fun = 0
    for line in node_line_sym:
        # 在行里查找所有 FUN 后面跟数字的模式
        for match in re.findall(r"\bFUN(\d+)\b", line):
            num = int(match)
            if num > max_fun:
                max_fun = num
    return max_fun


def count_control_structures(node_lines: List[str]) -> int:
    """
    统计 node_line 中 if/for/while 出现的总次数。
    """
    count = 0
    for line in node_lines:
        # 只要在行文本中出现关键字，即算一次。为了避免误匹配，可用正则。
        count += len(re.findall(r"\bif\b", line))
        count += len(re.findall(r"\bfor\b", line))
        count += len(re.findall(r"\bwhile\b", line))
    return count


def reconstruct_code(node_lines: List[str]) -> str:
    """
    将 node_line（字符串列表）拼接成一个完整的代码字符串，中间用换行分隔。
    """
    return "\n".join(node_lines)


def compute_sample_features(
    sample: Dict[str, Any],
    vectorizer: CodeVectorizer
) -> Dict[str, Any]:
    """
    对单个样本计算所需特征：
      - var_num: node_line_sym 中最大的 VAR<number>
      - fun_num: node_line 中函数调用数
      - key_num: node_line 中 if/for/while 数
      - l: node_line 的行数
      - V: 代码字符串的语义向量
    返回一个字典，包含原 sample 信息和新的特征。
    """
    node_lines: List[str] = sample["node_line"]
    node_line_sym: List[str] = sample["node_line_sym"]

    # 1. var_num
    var_num = extract_features_from_node_line_sym(node_line_sym)

    # 2. fun_num
    fun_num = count_function_calls(node_line_sym)

    # 3. key_num
    key_num = count_control_structures(node_lines)

    # 4. l
    l = len(node_lines)

    # 5. 拼接成代码字符串 S
    code_str = reconstruct_code(node_lines)

    # 6. 语义向量 V
    V = vectorizer.encode(code_str)  # numpy array

    return {
        "filepath": sample["filepath"],
        "label": sample["label"],
        "var_num": var_num,
        "fun_num": fun_num,
        "key_num": key_num,
        "l": l,
        "V": V,
        "original_sample": sample,
    }

# ------------------------------------------------------------------------------
# 2. 聚类与模型保存部分
# ------------------------------------------------------------------------------


class Clusterer:
    def __init__(self, n_clusters: int, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans: KMeans = None

    def fit(self, feature_matrix: np.ndarray):
        """
        对特征矩阵进行 KMeans 聚类，并保存模型。
        feature_matrix: shape = (num_samples, feature_dim)
        """
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        self.kmeans.fit(feature_matrix)

    def predict(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        对新的特征矩阵进行聚类预测，返回每个样本的簇标签。
        """
        return self.kmeans.predict(feature_matrix)

    def save(self, output_path: str):
        """将 KMeans 模型序列化保存到文件"""
        with open(output_path, "wb") as f:
            pickle.dump(self.kmeans, f)

    def load(self, input_path: str):
        """从文件中加载已经训练好的 KMeans 模型"""
        with open(input_path, "rb") as f:
            self.kmeans = pickle.load(f)

    def get_cluster_centers(self) -> np.ndarray:
        """获取聚类中心 (n_clusters, feature_dim)"""
        return self.kmeans.cluster_centers_

    def find_representative_samples(
        self, feature_matrix: np.ndarray
    ) -> List[int]:
        """
        对于给定的 feature_matrix，逐簇找到距离簇中心最近的样本索引。
        返回长度为 n_clusters 的列表，每个元素是该簇的“中心样本”在 feature_matrix 中的索引。
        """
        centers = self.get_cluster_centers()  # (n_clusters, feature_dim)
        labels = self.kmeans.labels_  # (num_samples,)
        reps = []
        for cluster_id in range(self.n_clusters):
            # 找到属于该簇的所有样本索引
            idxs = np.where(labels == cluster_id)[0]
            if len(idxs) == 0:
                # 该簇为空（不太可能，但保险起见）
                reps.append(None)
                continue
            # 计算这些样本到簇中心的距离
            dists = np.linalg.norm(
                feature_matrix[idxs] - centers[cluster_id], axis=1)
            # 找到距离最小的那个
            min_idx_in_cluster = idxs[np.argmin(dists)]
            reps.append(min_idx_in_cluster)
        return reps

# ------------------------------------------------------------------------------
# 3. 主流程示例
# ------------------------------------------------------------------------------


def main(
    input_json_path: str,
    output_dir: str,
    n_clusters_label0: int = 5,
    n_clusters_label1: int = 5,
    pretrained_model: str = "microsoft/codebert-base",
):
    """
    整个流程演示：
      1. 加载 JSON
      2. 特征提取 + 向量化
      3. 按 label 分组，分别聚类
      4. 提取聚类中心样本
      5. 保存模型
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载 JSON
    with open(input_json_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    # 2. 实例化代码向量化器
    vectorizer = CodeVectorizer(model_name=pretrained_model)

    # 3. 对所有样本做特征提取
    processed_samples = []
    print("开始特征提取和向量化，总样本数:", len(samples))
    for idx, samp in enumerate(samples):
        feat = compute_sample_features(samp, vectorizer)
        processed_samples.append(feat)
        if (idx + 1) % 50 == 0 or idx == len(samples) - 1:
            print(f"  已处理 {idx+1}/{len(samples)} 样本")

    # 4. 按 label 分组
    group0 = [p for p in processed_samples if p["label"] == 0]
    group1 = [p for p in processed_samples if p["label"] == 1]
    print(f"Label=0 样本数: {len(group0)}, Label=1 样本数: {len(group1)}")

    # 5. 分别构造特征矩阵并聚类
    def build_feature_matrix(group: List[Dict[str, Any]]) -> np.ndarray:
        """
        将 group 中每个样本 var_num, fun_num, key_num, l, V 拼接成一个特征向量。
        """
        mats = []
        for item in group:
            scalar_feats = np.array(
                [item["var_num"], item["fun_num"], item["key_num"], item["l"]], dtype=float)
            vec = item["V"]
            feat = np.concatenate([scalar_feats, vec], axis=0)
            mats.append(feat)
        return np.vstack(mats)  # shape = (num_samples_in_group, 4 + emb_dim)

    # ----- 对 Label=0 组进行聚类 -----
    feats0 = build_feature_matrix(group0)
    clusterer0 = Clusterer(n_clusters=n_clusters_label0)
    print("开始对 Label=0 组聚类...")
    clusterer0.fit(feats0)

    # 提取 Label=0 组的中心样本索引
    reps0_idxs = clusterer0.find_representative_samples(feats0)
    reps0 = [group0[i]["original_sample"] for i in reps0_idxs if i is not None]

    # 保存 Label=0 的 KMeans 模型
    model0_path = output_dir / f"kmeans_label0_{n_clusters_label0}.pkl"
    clusterer0.save(str(model0_path))
    print(f"Label=0 聚类模型保存在: {model0_path}")

    # 将中心样本写入 JSON 以供查看
    with open(output_dir / f"representatives_label0_{n_clusters_label0}.json", "w", encoding="utf-8") as f0:
        json.dump(reps0, f0, ensure_ascii=False, indent=2)

    # ----- 对 Label=1 组进行聚类 -----
    feats1 = build_feature_matrix(group1)
    clusterer1 = Clusterer(n_clusters=n_clusters_label1)
    print("开始对 Label=1 组聚类...")
    clusterer1.fit(feats1)

    # 提取 Label=1 组的中心样本索引
    reps1_idxs = clusterer1.find_representative_samples(feats1)
    reps1 = [group1[i]["original_sample"] for i in reps1_idxs if i is not None]

    # 保存 Label=1 的 KMeans 模型
    model1_path = output_dir / f"kmeans_label1_{n_clusters_label1}.pkl"
    clusterer1.save(str(model1_path))
    print(f"Label=1 聚类模型保存在: {model1_path}")

    # 将中心样本写入 JSON 以供查看
    with open(output_dir / f"representatives_label1_{n_clusters_label0}.json", "w", encoding="utf-8") as f1:
        json.dump(reps1, f1, ensure_ascii=False, indent=2)

    print("所有流程已完成。")

# ------------------------------------------------------------------------------
# 4. 对新 node_line 进行预测的函数
# ------------------------------------------------------------------------------


class TwoGroupPredictor:
    """
    加载之前保存的两个聚类模型，对新的 node_line 输出在两个组中的聚类类别。
    """

    def __init__(
        self,
        model0_path: str,
        model1_path: str,
        pretrained_model: str = "microsoft/codebert-base"
    ):
        # 加载向量化器
        self.vectorizer = CodeVectorizer(model_name=pretrained_model)

        # 加载两个集群模型
        self.clusterer0 = Clusterer(n_clusters=1)  # 只是占位，后面会替换 kmeans
        self.clusterer0.load(model0_path)

        self.clusterer1 = Clusterer(n_clusters=1)
        self.clusterer1.load(model1_path)

    def _compute_features_from_node_line(self, node_lines: List[str], node_line_sym: List[str]) -> np.ndarray:
        """
        利用与训练时一致的方式，从 node_lines + node_line_sym 中提取 var_num, fun_num, key_num, l,
        并通过预训练模型得到 V，然后拼接成完整特征向量返回。
        """
        var_num = extract_features_from_node_line_sym(node_line_sym)
        fun_num = count_function_calls(node_line_sym)
        key_num = count_control_structures(node_lines)
        l = len(node_lines)
        code_str = reconstruct_code(node_lines)
        V = self.vectorizer.encode(code_str)

        scalar_feats = np.array([var_num, fun_num, key_num, l], dtype=float)
        full_feat = np.concatenate([scalar_feats, V], axis=0)  # (4 + emb_dim,)
        return full_feat

    def predict_both_groups(
        self,
        node_lines: List[str],
        node_line_sym: List[str]
    ) -> Dict[str, int]:
        """
        输入 node_lines 和 node_line_sym，返回两组在各自模型中的簇标签：
            {
              "label0_cluster": int,
              "label1_cluster": int
            }
        """
        feat = self._compute_features_from_node_line(
            node_lines, node_line_sym).reshape(1, -1)
        label0_clu = self.clusterer0.predict(feat)[0]
        label1_clu = self.clusterer1.predict(feat)[0]
        return {
            "label0_cluster": int(label0_clu),
            "label1_cluster": int(label1_clu)
        }


# ------------------------------------------------------------------------------
# 5. 脚本可执行接口
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    """
    运行示例：
      python clustering_script.py \
        --input_json data/samples.json \
        --output_dir output_models \
        --n0 5 --n1 7
    """

    input_json = 'train.json'
    output_dir = '***'
    n = 15

    model_name = "microsoft/codebert-base"
    main(
        input_json_path=input_json,
        output_dir=output_dir,
        n_clusters_label0=n,
        n_clusters_label1=n,
        pretrained_model=model_name,
    )

    print("\n示例运行完毕。")

    n = 25

    model_name = "microsoft/codebert-base"
    main(
        input_json_path=input_json,
        output_dir=output_dir,
        n_clusters_label0=n,
        n_clusters_label1=n,
        pretrained_model=model_name,
    )

    print("\n示例运行完毕。")
