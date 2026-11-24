import json
import os

src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
json_path = os.path.join(src_dir, 'voice_results', 'parameters_experiment.json')

with open(json_path, encoding='utf-8') as f:
    data = json.load(f)

columns = []
for group, prefix in [('temperature', 't'), ('beam_size', 'beam')]:
    for value in next(iter(data.values()))[group]:
        columns.append((group, value, f'{prefix}={value}'))

name_width = max(len('Файл'), *(len(name) for name in data)) + 2
col_width = 9

header = f"{'Файл':<{name_width}}" + ''.join(f'{title:>{col_width}}' for _, _, title in columns)
print(header)
for file_name, res in data.items():
    row = f'{file_name:<{name_width}}'
    for group, value, _ in columns:
        text = res[group].get(value)
        row += f"{(len(text) if text is not None else '-'):>{col_width}}"
    print(row)
