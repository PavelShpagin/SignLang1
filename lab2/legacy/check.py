import nbformat
with open('lab2/lab2.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

with open('lab2/check.txt', 'w', encoding='utf-8') as out:
    for i, c in enumerate(nb.cells):
        if c.cell_type == 'code':
            if 'def get_finger_states' in c.source:
                out.write(f'--- Cell {i} has get_finger_states ---\n')
                out.write(c.source[:500] + '\n\n')
            if 'def run_gesture_recognition' in c.source:
                out.write(f'--- Cell {i} has run_gesture_recognition ---\n')
                out.write(c.source[:500] + '\n\n')
