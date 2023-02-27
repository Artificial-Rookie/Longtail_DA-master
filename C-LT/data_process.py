import json
import copy
import numpy as np

def count_data(path="/home/chengru/github/Longtail_DA-master/bdd100k_ori/train_day.json"):
    with open(path) as js:
        temp = json.load(js)
        images = temp["images"]
        annot = temp["annotations"]
        img2id = dict()
    for pair in images:
        img2id[pair["file_name"]] = pair["id"]  # use img name to get img id

    num_each = [0 for _ in range(10)]   # record the number of instances in each classes (in current json file)
    for temp in annot:
        num_each[temp["category_id"]-1] += 1    # count the numbers of instances in each class

    keep_rate = dict()
    min_num = max(num_each)
    for id in range(len(num_each)):
        if num_each[id] >= 3000:
            if num_each[id] < min_num:
                min_num = num_each[id]  # find the smallest class over 3000
    for id in range(len(num_each)):
        if num_each[id] < min_num:
            keep_rate[id] = 1
        else:
            keep_rate[id] = min_num/num_each[id]   # records the keep rate of each class in source data
            
    return keep_rate, num_each  # keep_rate keys starts from 1 to 10


def curation(train_dataset, img_num_list, num_classes, keep_rate):
    "curate the dataset to delete some of the samples to make it balance"
    data_list_val = {}
    for j in range(num_classes):
        data_list_val[j] = [i for i, label in enumerate(train_dataset.targets) if label == j]

    idx_to_meta = []
    idx_to_train = []
    print(img_num_list)

    for cls_idx, img_id_list in data_list_val.items():
        np.random.shuffle(img_id_list)
        img_num = img_num_list[int(cls_idx)]
        keep_len = round(keep_rate[cls_idx]*img_num)
        idx_to_meta.extend(img_id_list[:keep_len])
        idx_to_train.extend(img_id_list[keep_len:])

    imbalanced_train_data = copy.deepcopy(train_dataset)
    imbalanced_train_data.samples = np.delete(train_dataset.samples,idx_to_train,axis=0)
    imbalanced_train_data.targets = np.delete(train_dataset.targets, idx_to_train, axis=0)

    return imbalanced_train_data