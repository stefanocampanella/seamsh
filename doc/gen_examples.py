"""Converts the example scripts (../examples/*.py) into reStructuredText pages (examples/*.rst).

In the scripts, a line "# %%" starts a block of text, written as comments and ended by an empty
line; everything else is code. Leading SPDX lines (license header) and the empty line after them
are left out.
"""

import glob
import os
import shutil

shutil.rmtree("examples", ignore_errors=True)
os.makedirs("examples")

for filename in glob.glob("../examples/*.py"):
    inrst = True
    with open(filename) as f:
        lines = f.readlines()
    # Leave out the license header (SPDX lines) and the empty line after it.
    if lines and lines[0].startswith("# SPDX"):
        while lines and lines[0].startswith("# SPDX"):
            lines.pop(0)
        if lines and not lines[0].strip():
            lines.pop(0)
    bname = os.path.basename(filename)[:-3]
    with open("examples/" + bname + ".rst", "w") as fout:
        fout.write(f"Download :download:`this example <../{filename}>`.\n")
        for i, line in enumerate(lines):
            line = line[:-1]
            lstrip = line.strip()
            if lstrip == "# %%":
                inrst = True
                fout.write("\n")
                continue
            if (not lstrip and inrst) or i == 0 or (inrst and lstrip[0] != "#"):
                fout.write("\n.. code-block:: python\n  \n")
                inrst = False
                continue
            if inrst:
                fout.write(line[2:] + "\n")
            else:
                fout.write("  " + line + "\n")
