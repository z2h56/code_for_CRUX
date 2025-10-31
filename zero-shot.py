
from dataprocess import predata, dataformat
import numpy as np
import tqdm
from utils import * 

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



inference_model_name = 'llama3.1:8b' 

#'''
import time
from sklearn.metrics import f1_score,precision_score,recall_score
for idx in [4,5,6,7]:
    pred_labels = []
    test_instances = test_sets[idx][:1000]
    for test_instance in tqdm.tqdm(test_instances):
        query = f'{get_task_desc(test_instance)}\n{std_instance(test_instance)}'
        background = get_background(test_instance)
        #template = f'{query}'
        template = f'{background}\n#Question#: {query}'
        
        ans = get_ans_stream(template,model_name=inference_model_name)
        pred_labels.append(get_pred_label_stream(ans,inference_model_name))
    labels = [k.label for k in test_instances]
    print(f'{inference_model_name}-Dataset{idx}',precision_score(labels,pred_labels),recall_score(labels,pred_labels),f1_score(labels,pred_labels))
