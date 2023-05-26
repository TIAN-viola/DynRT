import pickle
import json

def load_file(filename):
    with open(filename, 'rb') as filehandle:
        ret = pickle.load(filehandle)
        return ret



if __name__ == '__main__':

    test_id = load_file('input/prepared/test_id')
    test_text = load_file('input/prepared/test_text')
    test_labels = load_file('input/prepared/test_labels')

    valid_id = load_file('input/prepared/valid_id')
    valid_text = load_file('input/prepared/valid_text')
    valid_labels = load_file('input/prepared/valid_labels')
 
    train_id = load_file('input/prepared/train_id')
    train_text = load_file('input/prepared/train_text')
    train_labels = load_file('input/prepared/train_labels')

    test_json = json.load(open('HKEmodel_dataset/test.json', 'r', encoding='utf-8'))
    valid_json = json.load(open('HKEmodel_dataset/val.json', 'r', encoding='utf-8'))
    train_json = json.load(open('HKEmodel_dataset/train.json', 'r', encoding='utf-8'))

    test_id_new = []
    test_text_new = []
    test_labels_new = []

    for i, line in enumerate(test_text):
        if "sarcasm" in line or "sarcastic" in line or "reposting" in line or "<url>" in line or "joke" in line or "humour" in line or "humor" in line or "jokes" in line or "irony" in line or "ironic" in line or "exgag" in line:
            continue
        else:
            test_text_new.append(line)
            test_id_new.append(test_id[i])
            test_labels_new.append(test_labels[i])
    
    pickle.dump(test_text_new, open('input/prepared_clean/test_text', 'wb'))
    pickle.dump(test_id_new, open('input/prepared_clean/test_id', 'wb'))
    pickle.dump(test_labels_new, open('input/prepared_clean/test_labels', 'wb'))

    if len(test_json) == len(test_text_new):
        print('test right!')

    train_id_new = []
    train_text_new = []
    train_labels_new = []

    for i, line in enumerate(train_text):
        if "sarcasm" in line or "sarcastic" in line or "reposting" in line or "<url>" in line or "joke" in line or "humour" in line or "humor" in line or "jokes" in line or "irony" in line or "ironic" in line or "exgag" in line:
            continue
        else:
            train_text_new.append(line)
            train_id_new.append(train_id[i])
            train_labels_new.append(train_labels[i])
    
    pickle.dump(train_text_new, open('input/prepared_clean/train_text', 'wb'))
    pickle.dump(train_id_new, open('input/prepared_clean/train_id', 'wb'))
    pickle.dump(train_labels_new, open('input/prepared_clean/train_labels', 'wb'))

    if len(train_json) == len(train_text_new):
        print('train right!')

    valid_id_new = []
    valid_text_new = []
    valid_labels_new = []

    for i, line in enumerate(valid_text):
        if "sarcasm" in line or "sarcastic" in line or "reposting" in line or "<url>" in line or "joke" in line or "humour" in line or "humor" in line or "jokes" in line or "irony" in line or "ironic" in line or "exgag" in line:
            continue
        else:
            valid_text_new.append(line)
            valid_id_new.append(valid_id[i])
            valid_labels_new.append(valid_labels[i])
    
    pickle.dump(valid_text_new, open('input/prepared_clean/valid_text', 'wb'))
    pickle.dump(valid_id_new, open('input/prepared_clean/valid_id', 'wb'))
    pickle.dump(valid_labels_new, open('input/prepared_clean/valid_labels', 'wb'))


    if len(valid_json) == len(valid_text_new):
        print('valid right!')