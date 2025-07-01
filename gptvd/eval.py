import json
# from clusters import TwoGroupPredictor
from gptvd.clusters import TwoGroupPredictor
from llm import Qwen_LLM


def main(to_eval_path, cluster_model_path, llm_model, api_key="your-key-here"
         ):
    predictor = TwoGroupPredictor(
        model0_path=f"***/kmeans_label0_{cluster_model_path}.pkl",
        model1_path=f"***/kmeans_label1_{cluster_model_path}.pkl",
        pretrained_model="microsoft/codebert-base"
    )

    with open(f"***/representatives_label0_{cluster_model_path}.json") as f:
        repentatives_label0 = json.load(f)

    with open(f"***/representatives_label1_{cluster_model_path}.json") as f:
        repentatives_label1 = json.load(f)

    with open(to_eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    tp, tn, fp, fn = 0, 0, 0, 0
    for i, e in enumerate(eval_data):
        cluster_result = predictor.predict_both_groups(
            e['node_line'], e['node_line_sym'])

        example_label_0 = repentatives_label0[cluster_result['label0_cluster']]
        example_label_1 = repentatives_label1[cluster_result['label1_cluster']]
        code = '\n'.join(e['node_line'])
        code_0 = '\n'.join(example_label_0['node_line'])
        code_1 = '\n'.join(example_label_1['node_line'])
        result_0 = example_label_0['result']
        result_1 = example_label_1['result']

        prompt = f'''# I will provide you with a slice of C++ code. Please analyze whether there is a vulnerability that will definitely be triggered.
## You should strictly follow the following instructions:
```instructions
1. This slice is incomplete. Therefore, if you encounter any missing definitions or uninitialized variables, please regard them as defined or initialized. At the same time, ignore derivative vulnerabilities caused by missing definitions or uninitialized.
2. If whether a vulnerability is triggered depends on an unknown variable value, an unknown function, or incomplete context, it should generally not be considered a definite vulnerability. However, for the following categories, perform detailed analysis:
    2.1 External-input variables (e.g., variables that originate from unknown sources, user input, or untrusted functions):        
        - When used as a source (e.g., for indexing, arithmetic, dereference, size calculations, function ): analyze whether its value could lead to buffer overflows, out-of-bounds accesses, injection issues, or logic errors. If such risks are possible based on type limits or code logic, mark as a potential vulnerability. 
    2.2 Arithmetic operations (integer and pointer arithmetic):
        - Even if operand values or pointer bases are unknown, examine whether operations could cause integer overflow/underflow or produce invalid pointer values (e.g., pointer arithmetic beyond object bounds, negative or wraparound results, misaligned access).
        - If the result depends on external or unknown input (e.g., return value of a function, runtime input, unbounded variable) and could exceed type limits or object boundaries, treat it as a vulnerability unless there are visible, effective runtime checks that definitively prevent it.
        - Do not rely solely on `assert` or similar debug-only checks: if no guaranteed-enforced check is visible, assume the worst case.
3. Only analyze the currently visible code logic, without speculating on potential issues or possible vulnerabilities.
```
## You must analyze according to the following format:
```format
Analysis:
    [line 1]: [logic of the code]
    [Is this a special case?] [yes/no]
    [reason for the vulnerability judgment]
    [vulnerability: yes/no]
    [line 2]: [logic of the code]
    [Is this a special case?] [yes/no]
    [reason for the vulnerability judgment]
    [vulnerability: yes/no]
    [line n]: [logic of the code]
    [Is this a special case?] [yes/no]
    [reason for the vulnerability judgment]
    [vulnerability: yes/no]
Final_result: YES / Final_result: NO
```
## Next is the code slice to be analyzed:
```c++
{code}
```
## Now, please begin your analysis.'''

        llm = Qwen_LLM(api_key, llm_model)
        result = llm.generate(prompt)
        # 将result保存到e['pred']中
        e['pred'] = result
        # print(result+f'\n{i+1}\n')
        parse_result = result.split("Final_result:")[-1].strip()
        # 将parse_result转成小写
        parse_result = parse_result.lower()[:10]
        if 'no' in parse_result:
            parse_result = 0  # 不存在漏洞
        elif 'yes' in parse_result:
            parse_result = 1  # 存在漏洞
        else:
            print(f"Unexpected result: {parse_result}")
            print(f"Unexpected result: {parse_result}")
            print(f"Unexpected result: {parse_result}")
            continue

        true_label = e['label']
        if true_label == 0:
            if parse_result == 0:
                tn += 1
            else:
                fp += 1
        elif true_label == 1:
            if parse_result == 1:
                tp += 1
            else:
                fn += 1
                continue

        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        print(f"Accuracy: {(tp + tn) / (tp + tn + fp + fn):.4f}")
        print(f"Precision: {tp / (tp + fp) if (tp + fp) > 0 else 0:.4f}")
        print(f"Recall: {tp / (tp + fn) if (tp + fn) > 0 else 0:.4f}")
        print(
            f"F1 Score: {2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0:.4f}")

        # 将eval_data覆盖原文件
        with open(to_eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, indent=4)

    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"Accuracy: {(tp + tn) / (tp + tn + fp + fn):.4f}")
    print(f"Precision: {tp / (tp + fp) if (tp + fp) > 0 else 0:.4f}")
    print(f"Recall: {tp / (tp + fn) if (tp + fn) > 0 else 0:.4f}")
    print(
        f"F1 Score: {2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0:.4f}")


cluster_model_path = 10


to_eval_path = "***.json"
llm_model = "qwen-plus-latest"
main(to_eval_path, cluster_model_path, llm_model)
