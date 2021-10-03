import json

def get_data_from_json(file_path):
    with open(file_path, encoding="utf8") as json_file:
        data = json.load(json_file)
        
    return data

def get_text_and_label(wipo_dict, selected_cases_file_path=None):
    cases = dict()
    labels = dict()
    label_transfer = []
    label_denied = []
    label_cancel = []

    selected_cases = set()

    if selected_cases_file_path:
        with open(selected_cases_file_path) as fp:
            lines = fp.readlines()
            for line in lines:
                selected_cases.add(line.strip())

    for key, val in wipo_dict.items():
        if selected_cases_file_path:
            if key not in selected_cases:
                continue
            
        cases[key] = val['text']
        labels[key] = val['status']

        if val['status']  == 'transfer':
            label_transfer.append(key)
        elif val['status'] == 'complaint denied':
            label_denied.append(key)
        elif val['status'] == 'cancellation':
            label_cancel.append(key)

    return cases, labels, label_transfer, label_denied, label_cancel
