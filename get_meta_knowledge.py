from dataprocess import predata, dataformat
import numpy as np
from utils import * 
from sentence_transformers import SentenceTransformer, util
import json

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import networkx as nx
from collections import defaultdict
import community as community_louvain  # pip install python-louvain


train_sets = []
test_sets = []
valid_sets = []
limit = 10000
metrics = []

for key,p in dataformat.new_deepmatcher_data.items():    #4
    train_sets.append(data2instances(get_data(p[1]+"train.json",num=limit),'ER',p[1]))
    test_sets.append(data2instances(get_data(p[1]+"test.json"),'ER',p[1]))
    valid_sets.append(data2instances(get_data(p[1]+"valid.json",num=limit),'ER',p[1]))
    metrics.append(p[2])

for key,p in dataformat.entity_linking_data.items():   #2
    train_sets.append(data2instances(get_data(p[1]+"train.json",num=limit),'EL',p[1]))
    test_sets.append(data2instances(get_data(p[1]+"test.json"),'EL',p[1]))
    valid_sets.append(data2instances(get_data(p[1]+"valid.json",num=limit),'EL',p[1]))
    metrics.append(p[2])


for key,p in dataformat.entity_alignment_data.items():    #2
    train_sets.append(data2instances(get_data(p[1]+"train.json",num=limit),'EA',p[1]))
    test_sets.append(data2instances(get_data(p[1]+"test.json"),'EA',p[1]))
    valid_sets.append(data2instances(get_data(p[1]+"valid.json",num=limit),'EA',p[1]))
    metrics.append(p[2])


encoder = SentenceTransformer('./all-MiniLM-L6-v2')
knowledge_model_name = 'llama3.1-8b'

demo_size = 700


rule_embeddings = encoder.encode(er_rules, 
                               convert_to_tensor=True,  
                               show_progress_bar=True)  

embeddings_np = rule_embeddings.cpu().numpy()


def graph_guided_kmeans_grouped_rules_louvain(l, encoder, top_k=2, kmeans_n_init=10, random_state=42, subcluster_ratio=0.1):
    embeddings = encoder.encode(l, convert_to_tensor=True, show_progress_bar=True)
    embeddings_np = embeddings.cpu().numpy()

    sim_matrix = cosine_similarity(embeddings_np)
    G = nx.Graph()
    G.add_nodes_from(range(len(embeddings_np)))
    for i in range(len(embeddings_np)):
        top_indices = np.argsort(sim_matrix[i])[::-1][1:top_k+1]
        for j in top_indices:
            G.add_edge(i, j, weight=sim_matrix[i][j])

    partition = community_louvain.best_partition(G, weight='weight')
    init_labels = np.array([partition[i] for i in range(len(embeddings_np))])

    final_labels = np.zeros(len(embeddings_np), dtype=int)
    cluster_id = 0
    for community_id in sorted(set(init_labels)):
        indices = np.where(init_labels == community_id)[0]
        sub_embeddings = embeddings_np[indices]

        if subcluster_ratio is not None and len(indices) > 1:
            K_m = max(1, int(len(indices) * subcluster_ratio))
        else:
            K_m = 1

        if K_m > 1:
            kmeans = KMeans(
                n_clusters=K_m,
                init='k-means++',
                n_init=kmeans_n_init,
                random_state=random_state
            )
            kmeans.fit(sub_embeddings)
            for idx, sub_label in zip(indices, kmeans.labels_):
                final_labels[idx] = cluster_id + sub_label
            cluster_id += K_m
        else:
            final_labels[indices] = cluster_id
            cluster_id += 1

    cluster_dict = defaultdict(list)
    for idx, label in enumerate(final_labels):
        cluster_dict[label].append(l[idx])

    clustered_rules = list(cluster_dict.values())
    return clustered_rules

result = graph_guided_kmeans_grouped_rules_louvain(er_rules, encoder)

meta_rules = []
for i in range(len(result)):
    rules = '\n'.join(result[i])
    tmp = f"""Below is a list of rules.
    {rules}
    Please analyze these rules and extract a high-quality matching **meta rule**. The meta rule should capture the shared pattern behind all rules. Directly output it without other words.
    """
    # completion = openai.ChatCompletion.create(model=knowledge_model_name, messages=[{"role": "system", "content": "You are an expert in entity matching and knowledge extraction."},{"role": "user", "content": tmp}],temperature = 0.1,stream = False)
    # answer = completion.choices[0].message.content
    # meta_rules.append(answer.split('\n')[-1])
    completion = ollama.chat(model='llama3.1:8b',stream = False, messages=[{"role": "system", "content": "You are an expert in entity matching and knowledge extraction."},{"role": "user", "content": tmp}],options={"temperature":0.1})
    meta_rules.append(completion.message.content)

meta_rules_save_path = f'rules/meta_rules/{knowledge_model_name}_balancedAL_demosize{demo_size}_graph_guided_kmeans.json'
import json
with open(meta_rules_save_path, 'w') as f:
   json.dump(meta_rules, f) 


def trasnfer_task_rules(er_rules, task, model_name = 'llama3.1:8b'):
    task_rules = []
    for er_rule in er_rules:
        if task == 'EL':
            q = f'This is a meta rule for entity resolution task.\n{er_rule}. According to this, write a rule for entity linking task, which aims to determine whether a piece of text is related to the given entity.'
        elif task == 'EA':
             q = f'This is a meta rule for entity resolution task.\n{er_rule}. According to this, write a rule for entity alignment task, which aims to determine whether two KG-entities are the same real-world object.'
        try:
            a = get_ans(q,model_name,0.1)
            task_rules.append(a.split('\n')[-1])
        except:
            pass
    return task_rules

target_task = 'EL'
meta_task_rules = trasnfer_task_rules(meta_rules,target_task)


meta_task_rules_save_path = f'rules/meta_task_rules/{target_task}/{knowledge_model_name}_balancedAL_demosize{demo_size}_graph_guided_kmeans.json'

import json
with open(meta_task_rules_save_path, 'w') as f:
    json.dump(meta_task_rules, f) 