from dataprocess import predata, dataformat
import numpy as np
from sentence_transformers import SentenceTransformer, util
import tqdm
import json
import time

from utils import * 

train_sets = []
test_sets = []
valid_sets = []
limit = 10000
metrics = []
encoder = SentenceTransformer('./all-MiniLM-L6-v2')

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


def retrieve_knowledge(query, knowledge_list,knowledge_embeddings, model, top_k=1):
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, knowledge_embeddings)[0]
    top_k_indices = similarities.argsort(descending=True)[:top_k]
    retrieved_knowledge = [knowledge_list[idx] for idx in top_k_indices]
    return '\n'.join(retrieved_knowledge)


with open('rules/meta_task_rules/EA/gpt-3.5-turbo-ca_balancedAL_demosize700_graph_guided_kmeans_plus_v2.json', 'r') as f:
    meta_task_rules = json.load(f)

meta_task_rules = [r.split(":")[-1] for r in meta_task_rules]
meta_task_rules_embeddings = encoder.encode(meta_task_rules, convert_to_tensor=True)

test_instances = test_sets[6]                                                                                           
inference_model_name = "llama3.1:8b"
pred_labels = []


for test_instance in tqdm.tqdm(test_instances):
    query = f'{get_task_desc(test_instance)}\n{std_instance(test_instance)}'
    knowledge = retrieve_knowledge(query, meta_task_rules, meta_task_rules_embeddings, encoder,2)
    background = get_background(test_instance)
    template = f'{background} You can refer to the rules below.\n#Rules#: {knowledge} \n#Question#: {query}'
    #template = f'{background}\n#Question#: {query}\nYou can refer to the rules below.\n#Rules#: {knowledge}'
    ans = get_ans_stream(template,model_name=inference_model_name)
    pred_labels.append(get_pred_label_stream(ans,inference_model_name))
    
labels = [k.label for k in test_instances]
from sklearn.metrics import f1_score,precision_score,recall_score
print(precision_score(labels,pred_labels),recall_score(labels,pred_labels),f1_score(labels,pred_labels))