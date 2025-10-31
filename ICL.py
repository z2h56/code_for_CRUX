from dataprocess import predata, dataformat
import numpy as np 
import tqdm
from utils import *
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('./all-MiniLM-L6-v2')

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


demo_instance_space,test_instances = get_active_training_set(train_sets[:4],model,500),test_sets[7][:1000]
demo_space = [instance.serialized_pairs for instance in demo_instance_space]
pair1_embedding = model.encode([instance.pair1 for instance in demo_instance_space])
pair2_embedding = model.encode([instance.pair2 for instance in demo_instance_space])
train_knowledge_embedding = pair2_embedding-pair1_embedding
sentence_embeddings = model.encode(demo_space, convert_to_tensor=True)

model_name = 'llama3.1:8b'

pred_labels = []
for test_instance in tqdm.tqdm(test_instances):
    idxs = get_topk_relevant_id(demo_instance_space,test_instance,model,sentence_embeddings,train_knowledge_embedding,4,sim = 'Semantic')
    template = get_template(demo_instance_space,idxs,test_instance)
    ans = get_ans_stream(template,model_name = model_name)
    pred_labels.append(get_pred_label_stream(ans,model_name))
labels = [k.label for k in test_instances]
from sklearn.metrics import f1_score,precision_score,recall_score
print(precision_score(labels,pred_labels),recall_score(labels,pred_labels),f1_score(labels,pred_labels))