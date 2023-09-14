import argparse
import numpy as np
import torch
import random 
from transformers import IdeficsForVisionText2Text, AutoProcessor
from datasets import load_dataset
import PIL

def newyorker_caption_contest_data():

    dset = load_dataset("jmhessel/newyorker_caption_contest", "explanation")

    res = {}
    for spl, spl_name in zip([dset['train'], dset['validation'], dset['test']],
                            ['train', 'val', 'test']):
        cur_spl = []
        for inst in list(spl):
            inp = inst['from_description']
            targ = inst['label']
            cur_spl.append({'input': inp, 'target': targ, 'instance_id': inst['instance_id'], 'image': inst['image'], 'caption_choices': inst['caption_choices']})
        
            #'input' is an image annotation we will use for a llama2 e.g. "scene: the living room description: A man and a woman are sitting on a couch. They are surrounded by numerous monkeys. uncanny: Monkeys are found in jungles or zoos, not in houses. entities: Monkey, Amazon_rainforest, Amazon_(company)."
            #'target': a human-written explanation 
            #'image': a PIL Image object
            #'caption_choices': is human-written explanation

        res[spl_name] = cur_spl
    return res

print("Loading data")
nyc_data = newyorker_caption_contest_data()
nyc_data_five_val = random.sample(nyc_data['val'],5)
nyc_data_train_two = random.sample(nyc_data['train'],2)
print(nyc_data_train_two[0]['image'].type)
