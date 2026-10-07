import sys
here = r"C:/Users/Ors/AppData/Local/Temp/claude/D--preference-destillation/ffe199e2-eaba-41f3-879f-8ba3f4ec1e4f/scratchpad/slideshow/"
template = open(here + "template.html", encoding="utf-8").read()
data = open(here + "pertoken_js.txt", encoding="utf-8").read()
assert template.count("/*PERTOKEN_DATA*/null") == 1
data = data.replace("</", "<" + chr(92) + "/")
out = template.replace("/*PERTOKEN_DATA*/null", data)
target = sys.argv[1]
open(target, "w", encoding="utf-8", newline="\n").write(out)
print("wrote", target, len(out))
