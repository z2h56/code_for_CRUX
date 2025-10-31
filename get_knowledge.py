from dataprocess import predata, dataformat
import numpy as np
from utils import * 
from sentence_transformers import SentenceTransformer, util

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


for key,p in dataformat.string_matching_data.items():    #5
    train_sets.append(data2instances(get_data(p[1]+"train.json",num=limit),'StM',p[1]))
    test_sets.append(data2instances(get_data(p[1]+"test.json"),'StM',p[1]))
    valid_sets.append(data2instances(get_data(p[1]+"valid.json",num=limit),'StM',p[1]))
    metrics.append(p[2])

encoder = SentenceTransformer('./all-MiniLM-L6-v2')
knowledge_model_name = 'llama3.1:8b'
demo_size = 700
demo_instance_space = get_balanced_active_training_set(train_sets[:4],encoder,demo_size)

def get_reason_process(demo_instance_space,model_name = knowledge_model_name):
    
    reason_process, error_instance, error_process = [],[],[]
    #task_desc = get_task_desc(demo_instance_space[0])
    task_desc = "Do the two following entity descriptions refer to the same entity? Tell me yes or no and explain the reason."# 
    for test_instance in demo_instance_space:
        label = 'yes' if test_instance.label == 1 else 'no'
        r1_q = f"This is an entity resolution task. You need to judge whether two entity descriptions refer to the same entity.\n{task_desc}\n{std_instance(test_instance)}"
        try:
            r1_a = get_ans(r1_q,model_name,0.1)
        except:
            r1_a =''
        #knowledge evaluate
        if test_instance.label != get_pred_label(r1_a):
            error_instance.append(test_instance)
            error_process.append(r1_a)
            #reason_process.append('')
            try:
                if label == 'yes':
                    r1_q2 = f"Why do the two following entity descriptions refer to the same entity?\n{std_instance(test_instance)}\n"
                    r1_a2 = get_ans(r1_q2,model_name,0.1)
                else:
                    r1_q2 = f"Why do the two following entity descriptions refer to different entities?\n{std_instance(test_instance)}\n"
                    r1_a2 = get_ans(r1_q2,model_name,0.1)
            except:
                    r1_a2 =''
            reason_process.append(r1_a2)
            
            continue
        reason_process.append(r1_a)
    return reason_process

reason_process = get_reason_process(demo_instance_space)


def get_er_rules(demo_instance_space,reason_process,model_name = knowledge_model_name):
    er_rules = []
    task_desc = get_task_desc(demo_instance_space[0])
    for i in range(len(demo_instance_space)):
        test_instance = demo_instance_space[i]
        if reason_process[i] == '':
            er_rules.append('')
            continue
        r2_q = f"Given a reasoning process, please generate a rule in a natural language form to guide the determination of whether two entity descriptions refer to the same entity. The rule should be general and concise. You need to directly output it."+"\n"+ f"{std_instance(test_instance)}\n{reason_process[i]}"
        try:
            r2_a = get_ans(r2_q,model_name,0.01)
        except:
            r2_a = ''
        er_rules.append(r2_a)
    return er_rules

er_rules = get_er_rules(demo_instance_space,reason_process)
er_rules_save_path = f'rules/er_rules/{knowledge_model_name}_balancedAL_demosize{len(demo_instance_space)}.json'

import json
with open(er_rules_save_path, 'w') as f:
    json.dump(er_rules, f)
