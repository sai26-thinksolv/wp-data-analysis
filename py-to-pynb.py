
import nbformat as nbf

# Read .py file
with open("main.py") as f:
    code = f.read()

# Create a new notebook
nb = nbf.v4.new_notebook()
nb['cells'] = [nbf.v4.new_code_cell(code)]

# Write to ipynb
with open("domain-analyzer.ipynb", "w") as f:
    nbf.write(nb, f)
