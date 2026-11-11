import io, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
README = r"H:\llamacpp-c\README.md"
ARGPARSE = r"C:\Users\<user>\AppData\Local\Programs\Python\Python312\Lib\argparse.py"

readme_lines = io.open(README, encoding="utf-8").read().splitlines()
argparse_lines = io.open(ARGPARSE, encoding="utf-8").read().splitlines()

# real document text, no repetition-padding
doc_1000 = "\n".join(readme_lines[19:97])   # lines 20..97: quick start + description + backends table
doc_2500 = "\n".join(readme_lines[19:134]) + "\n\n" + "\n".join(argparse_lines[0:200])

q_1000 = (
    "Using only the llama.cpp README text above, answer in a numbered list. "
    "(1) List every backend from the 'Supported backends' table together with its target devices, "
    "in table order. (2) Name the three single-header libraries that are acknowledged for the "
    "multimodal subsystem. (3) State which backend targets 'IBM Z & LinuxONE' and which one targets "
    "'VirtGPU APIR'. Quote the exact text where possible."
)

q_2500 = (
    "Using the llama.cpp README and the Python argparse source above, answer precisely. "
    "(1) In the README, which backend targets 'IBM Z & LinuxONE', and which targets 'VirtGPU APIR'? "
    "(2) In argparse.py, what does the method _ActionsContainer.add_argument_group return, and what "
    "is the default value of its conflict_handler argument? Quote the exact source lines. "
    "(3) In argparse.py, quote the first paragraph of the ArgumentParser class docstring. "
    "(4) In the README, list the three tools whose documentation links appear under '#### Tools'."
)


def chat(user_text):
    return "<|im_start|>user\n" + user_text + "<|im_end|>\n<|im_start|>assistant\n"


def build(doc, question):
    return chat(doc + "\n\n" + question)


for name, text in [("p1000", build(doc_1000, q_1000)), ("p2500", build(doc_2500, q_2500))]:
    path = os.path.join(ROOT, name + ".txt")
    io.open(path, "w", encoding="utf-8", newline="\n").write(text)
    print(name, "chars:", len(text), "->", path, "doc_chars:", len(doc_1000) if name == "p1000" else len(doc_2500))
