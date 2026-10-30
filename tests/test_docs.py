"""Run the Python code of the README and of the documentation pages, and check that every
`cel.<function>` called in their Python code exists and accepts the keyword arguments used.

The README and the quick start page run exactly as written. The tutorial pages train for 500
epochs, so here every `num_epochs_max` is set to 3: this checks that their code runs, not their
numbers. Two tutorial pages are not run: the data augmentation page, because its autoencoder
trains for 449 epochs whatever the arguments, and the pretrained model page, because its query
data come from a 518 MB archive on Zenodo that is not part of the repository."""
import ast
import glob
import inspect
import os
import re

import pytest

import CeLEry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
# Pages to run, with the number of epochs that replaces num_epochs_max (None: run as written).
PAGES = {"README.md": None,
         "docs/get-started.md": None,
         "docs/tutorials/coordinates.md": 3,
         "docs/tutorials/layers.md": 3,
         "docs/tutorials/domains.md": 3}


def python_blocks(path):
    with open(path, encoding="utf-8") as f:
        return re.findall(r"```python[ \t]*\n(.*?)```", f.read(), flags=re.S)


@pytest.mark.parametrize("page", [p for p in PAGES if os.path.exists(os.path.join(ROOT, p))])
def test_page_runs(page, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    blocks = python_blocks(os.path.join(ROOT, page))
    assert blocks, page + " has no Python code"
    if PAGES[page] is not None:
        blocks = [re.sub(r"num_epochs_max\s*=\s*\d+", "num_epochs_max = %d" % PAGES[page], b) for b in blocks]
    monkeypatch.chdir(ROOT)   # the pages read data relative to the repository root and write to output/
    namespace = {"__name__": "__docs__"}
    for block in blocks:
        exec(compile(block, page, "exec"), namespace)
    import matplotlib.pyplot as plt
    plt.close("all")


def test_documented_calls_exist():
    pages = glob.glob(os.path.join(DOCS, "**", "*.md"), recursive=True) + [os.path.join(ROOT, "README.md")]
    problems = []
    for page in pages:
        for block in python_blocks(page):
            try:
                tree = ast.parse(block)
            except SyntaxError as err:
                problems.append("%s: %s" % (os.path.relpath(page, ROOT), err))
                continue
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name) and node.func.value.id == "cel"):
                    name = node.func.attr
                    where = "%s: cel.%s" % (os.path.relpath(page, ROOT), name)
                    if not hasattr(CeLEry, name):
                        problems.append(where + " does not exist")
                        continue
                    parameters = inspect.signature(getattr(CeLEry, name)).parameters
                    problems += ["%s has no argument %s" % (where, k.arg) for k in node.keywords
                                 if k.arg is not None and k.arg not in parameters]
    assert not problems, "\n".join(problems)
