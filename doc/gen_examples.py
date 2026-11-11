"""Converts the example scripts (../examples/*.py) into reStructuredText pages (examples/*.rst).

In the scripts, a line "# %%" starts a block of text, written as comments and ended by an empty
line; everything else is code.
"""

import glob
import os
import shutil

shutil.rmtree("examples", ignore_errors=True)
os.makedirs("examples")

for filename in glob.glob("../examples/*.py"):
    inrst = True
    with open(filename) as f:
        bname = os.path.basename(filename)[:-3]
        with open("examples/" + bname + ".rst", "w") as fout:
            fout.write(f"Download :download:`this example <../{filename}>`.\n")
            for i, line in enumerate(f):
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
