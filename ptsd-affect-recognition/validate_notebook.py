"""
validate_notebook.py — Prove the built notebook's inlined pipeline executes.

Extracts the "setup" code cell (which writes the inlined src/*.py to disk and
imports them), redirects PIPE to a local temp dir, and runs it + a synthetic
forward pass in the venv (which has torch/numpy/pandas/sklearn/matplotlib).
Also compiles every code cell to catch syntax errors in the embedded code.
"""
import json, os, subprocess, sys, tempfile, textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, "notebook", "PTSD_Affect_Recognition.ipynb")

with open(NB) as f:
    nb = json.load(f)

code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
print(f"Notebook: {len(nb['cells'])} cells, {len(code_cells)} code cells")

# 1. Syntax-compile every code cell (strip IPython `!`/`%` magics first)
for i, c in enumerate(code_cells):
    src = "".join(c["source"])
    # remove IPython shell/line magics which are not valid Python
    lines = [ln for ln in src.splitlines()
             if not ln.lstrip().startswith(("!", "%"))]
    try:
        compile("\n".join(lines), f"cell_{i}", "exec")
    except SyntaxError as e:
        print(f"[FAIL] cell {i} syntax: {e}")
        sys.exit(1)
print("[PASS] all code cells compile (IPython magics stripped)")

# 2. Extract the setup cell and smoke cell, run them with PIPE redirected
setup_src = None
smoke_src = None
for i, c in enumerate(code_cells):
    src = "".join(c["source"])
    if "PIPE = '/content/pipeline'" in src:
        setup_src = src
    if "SMOKE TEST PASSED" in src:
        smoke_src = src

assert setup_src is not None, "setup cell not found"
assert smoke_src is not None, "smoke cell not found"

tmp = tempfile.mkdtemp(prefix="pipe_")
setup_src = setup_src.replace("'/content/pipeline'", repr(tmp))

runner = textwrap.dedent(f"""
import sys
sys.path.insert(0, {HERE!r})
{setup_src}
{smoke_src.replace('!pip', '#!pip')}
""")

# strip the !pip / drive lines from requirements (not present in setup/smoke)
res = subprocess.run([sys.executable, "-c", runner],
                     cwd=HERE, capture_output=True, text=True)
print(res.stdout)
if res.returncode != 0:
    print("[FAIL] notebook inlined pipeline execution:\n", res.stderr[-3000:])
    sys.exit(1)

print("[PASS] inlined pipeline (setup + smoke) executed successfully")
print(f"[PASS] temp module dir: {tmp}")
