"""Print the results of an executed notebook: checks (✓ / PASS), errors, and the last lines of each output.

  python scripts/nb_report.py notebooks/04_thinking_off.ipynb [--all]

Used to read runs on Inara: ssh inara 'cd <repo> && python3 scripts/nb_report.py <notebook>'.
"""
import json
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
NOISE = ("Loading weights", "Resolving data files", "Warning: You are sending", "_pytree.py", "Failed to load")


def cell_text(output: dict) -> str:
    """Plain text of one output (stream, result, display, or error)."""
    if output["output_type"] == "error":
        return f"ERROR {output['ename']}: {output['evalue']}\n" + ANSI.sub("", "\n".join(output["traceback"][-8:]))
    text = output.get("text") or output.get("data", {}).get("text/plain", "")
    return ANSI.sub("", "".join(text) if isinstance(text, list) else text)


def main() -> None:
    path, show_all = sys.argv[1], "--all" in sys.argv
    cells = json.load(open(path))["cells"]
    ran = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and c.get("outputs")]
    print(f"{path}: {len(ran)} of {sum(c['cell_type'] == 'code' for c in cells)} code cells have output")
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "markdown":
            first = "".join(cell["source"]).strip().splitlines()
            if first and first[0].startswith("#"):
                print(f"\n{first[0]}")
            continue
        for output in cell.get("outputs", []):
            lines = [l for l in cell_text(output).splitlines() if l.strip() and not any(n in l for n in NOISE)]
            if not show_all:
                lines = lines[-12:]
            for line in lines:
                print(f"  [{i}] {line[:240]}")


if __name__ == "__main__":
    main()
