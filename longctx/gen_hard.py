import io, os

HERE = os.path.dirname(os.path.abspath(__file__))
README = r"H:\llamacpp-c\README.md"

readme_lines = io.open(README, encoding="utf-8").read().splitlines()
doc = "\n".join(readme_lines[19:97])          # exactly the same doc as p1000 (1146 tok)

q_hard = (
    "上面是 llama.cpp README 的一段英文文本。请完全用中文回答，并且不要照抄英文原文——"
    "要把关键信息翻译并改写成中文。\n"
    "(1) 文档中列出的每一个后端分别运行在什么目标设备/平台上？请按表格顺序逐一说明。\n"
    "(2) 多模态子系统致谢了哪三个单头库？\n"
    "(3) 表格里哪个后端面向 IBM Z 与 LinuxONE，哪个后端面向 VirtGPU APIR？"
)

prompt = "<|im_start|>user\n" + doc + "\n\n" + q_hard + "<|im_end|>\n<|im_start|>assistant\n"
path = os.path.join(HERE, "p_hard.txt")
io.open(path, "w", encoding="utf-8", newline="\n").write(prompt)
print("wrote", path, "chars", len(prompt))
