import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import PIL.Image as Image
import numpy as np

def get_obj(path="/home/chengru/github/Longtail_DA-master/bdd100k_ori/", out_dir="/home/chengru/github/Longtail_DA-master/bdd100k/val/"):
    with open(path+"val_day.json") as js:
        temp = json.load(js)
        images = temp["images"]
        annot = temp["annotations"]
        id2img = dict()

    for pair in images:
        id2img[pair["id"]] = pair["file_name"]  # use img name to get img id

    objects = []   # store the objects for each class
    for ann in annot:
        file_path = path+"test_day/"+id2img[ann["image_id"]]
        bbox = ann["bbox"]   # bounding box of the object
        cat = ann["category_id"]
        id = ann["id"]
        x, y, h, w = bbox
        x, y, h, w = map(float,[x,y,h,w])
        x1, x2, y1, y2 = map(round, [x,x+h,y,y+w])
        
        obj = {"id": id, "file_path": file_path, "bbox":(x1,x2,y1,y2), "cat": cat}
        objects.append(obj)

    for i in range(len(objects)):
        obj = objects[i]
        file_path = obj["file_path"]
        cls = obj['cat']
        x1,x2,y1,y2 = obj["bbox"]
        id = obj["id"]
        img = Image.open(file_path)
        im = img.crop((x1,y1,x2,y2))    # retransfrom the numpy array into Image type
        # im.save(out_dir+f'{i}.jpg')    # check the area
        image = im.resize((100,100))
        try:
            image.save(out_dir+str(cls)+f'/{id}.jpg')
        except:
            os.mkdir(out_dir+str(cls))
            image.save(out_dir+str(cls)+f'/{id}.jpg')


if __name__ == "__main__":
    get_obj()