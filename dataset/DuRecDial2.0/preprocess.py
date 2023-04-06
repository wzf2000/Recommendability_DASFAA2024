import json
from tqdm import tqdm

def get_data():
    return_data = []
    for phase in ['train', 'dev', 'test']:
        file_name = f'zh_{phase}.txt'
        with open(file_name, 'r') as f:
            lines = f.readlines()
            data_list = []
            for line in lines:
                data = json.loads(line)
                data_list.append(data)
        return_data.append(data_list)
    return return_data

def gen_entity(datas: list):
    entity_list = set()
    for data in datas:
        for line_data in tqdm(data):
            knowledge = line_data['knowledge']
            for edges in knowledge:
                for entity1, relation, entity2 in edges:
                    entity_list.add(entity1)
                    entity_list.add(entity2)
    entity2id = dict(zip(entity_list, range(1, 1 + len(entity_list))))
    with open('entity2id.json', 'w', encoding='utf-8') as f:
        json.dump(entity2id, f, ensure_ascii=False)

def main():
    datas = get_data()
    gen_entity(datas)

if __name__ == '__main__':
    main()