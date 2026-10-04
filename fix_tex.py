import os
import glob

def replace_in_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    if 'bmiranda' in content:
        content = content.replace('bmiranda', 'brando9')
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Updated {filepath}")

for filepath in glob.glob("paper_latex_and_notes/**/*.tex", recursive=True):
    replace_in_file(filepath)
