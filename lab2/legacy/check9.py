import nbformat
with open('lab2/lab2.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

with open('lab2/check9.txt', 'w', encoding='utf-8') as out:
    out.write(nb.cells[9].source)
