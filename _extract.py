import json, sys, os
sys.stdout.reconfigure(encoding='utf-8')

base = r'c:\Users\vasco\Desktop\Mestrado IA 1º\2º Semestre\A_Reforço'

files = [
    os.path.join(base, 'Aula_2', 'MDP_GridWorld_Incomplete.ipynb'),
    os.path.join(base, 'Aula_3', 'Practical3_Gridworld_INCOMPLETE.ipynb'),
    os.path.join(base, 'Aula_3', 'Practical3_CarRental_INCOMPLETE.ipynb'),
    os.path.join(base, 'Aula_03_03', 'Practical3_CarRental_INCOMPLETE.ipynb'),
]

for f in files:
    print(f"\n{'='*60}")
    print(f"FILE: {os.path.basename(f)}")
    print('='*60)
    nb = json.load(open(f, 'r', encoding='utf-8'))
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] == 'code':
            src = ''.join(c['source'])
            print(f"\n--- Cell {i} ---")
            print(src[:400])
