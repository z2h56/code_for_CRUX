class instance:
    def __init__(self,pairs,label,task,dataset):
        self.pairs = pairs
        self.serialized_pairs = '[SEP]'.join(pairs)
        self.pair1 = pairs[0]
        self.pair2 = pairs[1]
        self.task = task
        self.dataset = dataset
        self.label = int(label)
        self.pred_label = 0
    def __str__(self):
        return f'Pair1: {self.pair1}\nPair2: {self.pair2}\nLabel: {self.label}' 

def pairs2instance(l,task,dataset):
    return instance(l[:2],l[2],task,dataset)
def data2instances(data,task,dataset):
    res =[]
    for pairs in data:
        res.append(pairs2instance(pairs,task,dataset))
    return res


from sentence_transformers import SentenceTransformer, util
import ollama

import openai
openai.api_key = "xxxxxxx"
openai.api_base = "xxxxxx"

def get_data(filename,num=-1):
    data = json.load(open(filename,encoding='utf-8'))
    if num!=-1 and num<len(data):
        random.seed(42)
        data = random.sample(data,num)
    return data

def get_ans(template,model_name,temperature=0):
    if 'gpt' in model_name:
        completion = openai.ChatCompletion.create(model=model_name, messages=[{"role": "user", "content": template}],temperature = temperature,stream = False)
        return completion.choices[0].message.content
        
    else:
        completion = ollama.chat(model=model_name,stream = False, messages=[{"role": "user", "content": template}],options={"temperature":temperature})
        return completion.message.content


def get_ans_stream(template, model_name, temperature=0):
    if 'gpt' in model_name or 'gemini' in model_name or 'deepseek-v3' in model_name or 'kimi' in model_name or 'claude' in model_name:
        completion = openai.ChatCompletion.create(model=model_name, messages=[{"role": "user", "content": template}],temperature = temperature,stream = True)
    else:
        completion = ollama.chat(model=model_name,stream = True, messages=[{"role": "user", "content": template}],options={"temperature":temperature})    
    return completion

def get_pred_label(s):
    if 'yes' in s or 'Yes' in s or 'YES' in s:
        return 1
    return 0

def get_pred_label_s(s):
    if 'yes' in s or 'Yes' in s or 'YES' in s:
        return 1
    if 'no' in s or 'No' in s or 'NO' in s:
        return 0
    return None
    
def get_pred_label_stream(comp,model_name = 'qwen2.5:14b'):
    a = ''
    if 'gpt' in model_name or 'gemini' in model_name or 'claude' in model_name:
        for word in comp:
            tmp = word.choices[0].delta
            if tmp!={}:
                a += tmp.content
                t = get_pred_label_s(a)
                if t!=None:
                    return t
    else:
        for word in comp:
            a += word.message.content
            t = get_pred_label_s(a)
            if t!=None:
                return t
    #print(i,a)
    return t


def get_background(ins):
    if ins.task == 'ER':
        res = f'This is an entity resolution task. You need to judge whether two entity descriptions refer to the same entity.'
    elif ins.task == 'EL':
        res = f'This is an entity linking task. You need to judge whether a piece of text is related to the given entity.' 
    elif ins.task == 'EA':
        res = f'This is an entity aligment task. You need to judge whether two KG-entities are the same real-world object.' 

    return res

def get_task_desc(ins):
    if ins.task == 'ER':
        res = f'Do the two following entity descriptions refer to the same entity? Tell me yes or no.'
    elif ins.task == 'EL':
        res = f'Is the text related to the given entity? Tell me yes or no.' 
    elif ins.task == 'EA':
        res = f'Are the two following KG-entities the same real-world object? Tell me yes or no.' 
    return res

def std_instance(ins):
    if ins.task == 'ER':
        res = f'Entity1: {ins.pair1}\nEntity2: {ins.pair2}'
    elif ins.task == 'EL':
        res = f'Entity: {ins.pair1}\n Text:{ins.pair2}' 
    elif ins.task == 'EA':
        res = f' {ins.pair1}\nEntity2: {ins.pair2}'   
    return res


def get_demo(demo_instance_space,l):
    res = ''
    task_desc = get_task_desc(demo_instance_space[0])
    for i in range(len(l)):
        idx = l[i]
        label = 'Yes' if demo_instance_space[idx].label == 1 else 'No'
        res = res + f'#Example{i+1}#: {task_desc}\n{std_instance(demo_instance_space[idx])}\nAnswer: {label}\n'
    res = res.strip()
    return res



def semantic_similarities(query, sentence_embeddings,model) :
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, sentence_embeddings)
    return similarities.tolist()[0]


def get_topk_relevant_id(demo_instance_space,test_instance,model,sentence_embeddings,train_knowledge_embedding,k=2,sim = 'Jaccard',alpha = 0.7):
    demo_space = [instance.serialized_pairs for instance in demo_instance_space]
    target = test_instance.serialized_pairs
    if sim == 'Jaccard':
        similarities_with_id = [(i, jaccard_similarity(target, s)) for i, s in enumerate(demo_space)]
    elif sim == 'Semantic':
        similarities = semantic_similarities(target,sentence_embeddings,model)
        similarities_with_id = [(i, similarities[i]) for i in range(len(similarities))]
        
    similarities_with_id.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in similarities_with_id[:k]]




def get_template(demo_instance_space,idxs,test_instance):
    backgroud = get_background(test_instance)
    #backgroud_w_example = f'{backgroud} Here are some examples from other tasks and you can transfer some knowledge from them.'

    task_desc = get_task_desc(test_instance)
    question = f'#Question#: {task_desc}'
    demo = get_demo(demo_instance_space,idxs)
    template = f'{demo}\n{question}\n{std_instance(test_instance)}\nAnswer: '
    return template

def get_full_training_set(l):
    res = []
    for tmp in l:
        for ins in tmp:
            res.append(ins)
    return res
    
def distribute_integers(l, n):
    avg = l // n
    remainder = l % n
    result = [avg] * n
    for i in range(remainder):
        result[i] += 1
    return result



import torch
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

def k_center_greedy(sentence_embeddings, k):
    """
    使用 K-Center Greedy 算法选择样本。
    参数:
        sentence_embeddings (torch.Tensor): 句子嵌入向量，形状为 (n_samples, n_features)
        k (int): 需要选择的样本数量
    返回:
        selected_indices (list): 选择的样本索引列表
    """
    n_samples = sentence_embeddings.shape[0]
    embeddings_np = sentence_embeddings
    
    # 初始化选择的样本
    selected_indices = []
    remaining_indices = set(range(n_samples))
    
    # 随机选择第一个样本
    first_index = np.random.choice(list(remaining_indices))
    selected_indices.append(first_index)
    remaining_indices.remove(first_index)
    
    # 迭代选择剩余样本
    for _ in range(1, k):
        # 计算所有剩余样本到已选样本的最小距离
        distances = pairwise_distances(embeddings_np[list(remaining_indices)], 
                                       embeddings_np[selected_indices])
        min_distances = np.min(distances, axis=1)
        
        # 选择距离最远的样本
        farthest_index = list(remaining_indices)[np.argmax(min_distances)]
        selected_indices.append(farthest_index)
        remaining_indices.remove(farthest_index)
    
    return selected_indices



def get_active_training_set(l,model,budget = 500):   #l full train set
    res = []
    n = len(l)
    B = distribute_integers(budget,len(l))
    for i in range(len(l)):
        sentence_embeddings = model.encode([instance.serialized_pairs for instance in l[i]], convert_to_tensor=True).cpu().numpy()
        #idxs = active_learning_selection(sentence_embeddings, B[i])
        idxs = k_center_greedy(sentence_embeddings, B[i])
        for idx in idxs: 
            res.append(l[i][idx])
    return res



import random
from collections import defaultdict

def _safe_select_by_kcenter(model, samples, k):
    if k <= 0 or len(samples) == 0:
        return []
    k = min(k, len(samples))
    try:
        embs = model.encode(
            [ins.serialized_pairs for ins in samples],
            convert_to_tensor=True
        ).cpu().numpy()
        idxs = k_center_greedy(embs, k)
        if not idxs or len(idxs) < k:
            remain = [i for i in range(len(samples)) if i not in set(idxs or [])]
            random.shuffle(remain)
            idxs = (idxs or []) + remain[:(k - len(idxs or []))]
    except Exception:
        idxs = list(range(len(samples)))
        random.shuffle(idxs)
        idxs = idxs[:k]
    return [samples[i] for i in idxs]

def get_balanced_active_training_set(l, model, budget=500):
    """
    全局类别平衡（尽量 1:1）。如果某个数据集达不到 1:1，未用的名额（extra）
    会“跨数据集”转移给其他数据集，且以成对(0类+1类)的方式追加，保证全局平衡。
    """
    res = []
    n_bucket = len(l)
    B_list = distribute_integers(budget, n_bucket)

    # 按桶拆分 0/1 类样本
    buckets = []
    for bucket in l:
        s0 = [ins for ins in bucket if getattr(ins, "label", None) == 0]
        s1 = [ins for ins in bucket if getattr(ins, "label", None) == 1]
        buckets.append({"s0": s0, "s1": s1})

    chosen0 = [[] for _ in range(n_bucket)]
    chosen1 = [[] for _ in range(n_bucket)]

    # ---------- 第一轮：按 50/50 目标在每个桶内挑样 ----------
    total_extra = 0
    rem0_by_bucket = []
    rem1_by_bucket = []

    for i in range(n_bucket):
        s0, s1 = buckets[i]["s0"], buckets[i]["s1"]
        Bi = B_list[i]
        t0 = Bi // 2
        t1 = Bi - t0

        k0 = min(t0, len(s0))
        k1 = min(t1, len(s1))

        # 选中
        c0 = _safe_select_by_kcenter(model, s0, k0)
        c1 = _safe_select_by_kcenter(model, s1, k1)
        chosen0[i].extend(c0)
        chosen1[i].extend(c1)

        # 统计该桶未用的“额定名额” -> 形成全局 extra
        deficit = (t0 - k0) + (t1 - k1)
        total_extra += max(0, deficit)

        # 桶内剩余可用（第二轮可能追加）
        rem0 = [x for x in s0 if x not in set(c0)]
        rem1 = [x for x in s1 if x not in set(c1)]
        rem0_by_bucket.append(rem0)
        rem1_by_bucket.append(rem1)

    # 全局剩余额度以“成对”追加，保证全局 0/1 平衡
    # 每对=同时加入一个0类与一个1类样本
    total_pairs_need = total_extra // 2

    # 能追加的最大对数，受限于全局剩余 0/1 的可用量
    global_rem0 = sum(len(rem0_by_bucket[i]) for i in range(n_bucket))
    global_rem1 = sum(len(rem1_by_bucket[i]) for i in range(n_bucket))
    max_pairs_possible = min(total_pairs_need, global_rem0, global_rem1)

    pairs_left = max_pairs_possible

    # ---------- 第二轮：把对数分配给“仍有两类可用”的桶，桶内对称追加 ----------
    # 策略：优先给 “min(len(rem0_i), len(rem1_i))” 较大的桶（更能消化）
    bucket_order = sorted(
        range(n_bucket),
        key=lambda i: min(len(rem0_by_bucket[i]), len(rem1_by_bucket[i])),
        reverse=True
    )

    for i in bucket_order:
        if pairs_left <= 0:
            break
        cap_pairs_i = min(len(rem0_by_bucket[i]), len(rem1_by_bucket[i]))
        if cap_pairs_i <= 0:
            continue
        add_pairs = min(cap_pairs_i, pairs_left)

        # 从各自剩余里再挑 add_pairs 个（仍用 k-center 做代表性挑选）
        add0 = _safe_select_by_kcenter(model, rem0_by_bucket[i], add_pairs)
        # 从 rem 集合里去掉已选，避免重复
        rem0_by_bucket[i] = [x for x in rem0_by_bucket[i] if x not in set(add0)]

        add1 = _safe_select_by_kcenter(model, rem1_by_bucket[i], add_pairs)
        rem1_by_bucket[i] = [x for x in rem1_by_bucket[i] if x not in set(add1)]

        chosen0[i].extend(add0)
        chosen1[i].extend(add1)
        pairs_left -= add_pairs

    # 汇总结果
    for i in range(n_bucket):
        res.extend(chosen0[i])
        res.extend(chosen1[i])

    return res



