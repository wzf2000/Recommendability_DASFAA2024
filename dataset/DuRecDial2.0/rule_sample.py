import json
import random

filter_words_map = {

}
filter_words = filter_words_map.keys()

# Write a function that could give the type of the dialog according to the filter words
def check_type(dialog):
    if len(dialog) == 0:
        return 0
    for word in filter_words:
        for utterance in dialog:
            if word in utterance:
                return filter_words_map[word]
    return 0

# Write a function that could translate the following data into a json file with utf-8 encoding:
# File format for zh_*.txt and en_*.txt:
# Each line is json:
# {
#   "goal": "goal",
#   "user_profile": "user_profile",
#   "conversation": [
#     "user_utterance",
#     "system_utterance",
#     "user_utterance",
#     "system_utterance",
#     ...
#   ],
#   "other keys": ...
# }
# Expected output format for *zh_train.json*, *zh_dev.json*, and *zh_test.json*, en_* is the same:
# [
#   {
#     "goal": "goal",
#     "user_profile": "user_profile",
#     "conversation": [
#       "user_utterance",
#       "system_utterance",
#       "user_utterance",
#       "system_utterance",
#       ...
#     ],
#     "type": "type",
#     "other keys": ...
#   },
#   ...
# ]
def process(filename):
    with open(filename, 'r', encoding='utf8') as f:
        data = []
        for line in f:
            dialog = json.loads(line)
            dialog['type'] = check_type(dialog['conversation'])
            data.append(dialog)
    with open(filename.replace('.txt', '.json'), 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return data

# Write a function to random sample K dialogs from the *.json where the label is 0
def sample(filename, k=10):
    with open(filename, 'r', encoding='utf8') as f:
        data = json.load(f)
    data = [dialog for dialog in data if dialog['type'] == 0]
    random.shuffle(data)
    with open(filename.replace('.json', '_sample.json'), 'w', encoding='utf8') as f:
        json.dump(data[:k], f, ensure_ascii=False, indent=4)

# Write a function to call the above functions to process the data
def main():
    process('zh_train.txt')
    process('zh_dev.txt')
    process('zh_test.txt')
    process('en_train.txt')
    process('en_dev.txt')
    process('en_test.txt')
    sample('zh_train.json', 10)
    sample('zh_dev.json', 10)
    sample('zh_test.json', 10)
    sample('en_train.json', 10)
    sample('en_dev.json', 100)
    sample('en_test.json', 10)

if __name__ == '__main__':
    main()