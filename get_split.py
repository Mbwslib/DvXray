import os.path
import random
import json
import numpy as np

train_test_percent = 0.8

base_path = './data'
DvXray_set = [('DvXray', 'train')]

prohibited_item_classes = {'Gun': 0, 'Knife': 1, 'Wrench': 2, 'Pliers': 3, 'Scissors': 4, 'Lighter': 5, 'Battery': 6,
                           'Bat': 7, 'Razor_blade': 8, 'Saw_blade': 9, 'Fireworks': 10, 'Hammer': 11,
                           'Screwdriver': 12, 'Dart': 13, 'Pressure_vessel': 14}

def convert_annotation(image_id, list_file):

    with open('./data/DvXray/%s.json'%image_id, 'r', encoding='utf-8') as j:
        label = json.load(j)

    gt = np.zeros(15, dtype=int)
    objs = label['objects']
    if objs == 'None':
        gt = gt
    else:
        for obj in objs:
            ind = prohibited_item_classes[obj['label']]
            gt[ind] = 1

    list_file.write(' ' + ','.join([str(a) for a in gt]))



if __name__ == "__main__":

    random.seed(19)

    jsonfilepath = os.path.join(base_path, 'DvXray')
    saveBasePath = os.path.join(base_path, 'split')

    temp_file = os.listdir(jsonfilepath)

    total_json = []

    for js in temp_file:
        if js.endswith('.json'):
            total_json.append(js)

    num = len(total_json)
    list = range(num)
    tr = int(num * train_test_percent)
    train = random.sample(list, tr)

    print("train size", tr)

    ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
    ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')

    for i in list:
        name = total_json[i][:-5] + '\n'
        if i in train:
            ftrain.write(name)
        else:
            ftest.write(name)

    ftrain.close()
    ftest.close()

    for name, img_set in DvXray_set:
        image_ids = open(os.path.join(saveBasePath, '%s.txt'%img_set), encoding='utf-8').read().strip().split()
        list_file = open('%s_%s.txt'%(name, img_set), 'w', encoding='utf-8')

        for image_id in image_ids:
            list_file.write('./data/DvXray/%s_OL.png'%image_id)

            list_file.write(' ')

            list_file.write('./data/DvXray/%s_SD.png'%image_id)

            convert_annotation(image_id, list_file)

            list_file.write('\n')
        list_file.close()