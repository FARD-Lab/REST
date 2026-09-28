#!/usr/bin/env python3
"""Build the REST project page (index.html and static/) from the arXiv version of the paper.

The page shows the paper's main-text figures and tables exactly as the paper renders them,
in paper order, with the paper's numbers and captions. Figures are the paper's own PDFs;
tables are typeset by pdflatex with the paper's preamble, so both match the PDF.

Steps:
  1. Compile the paper so main.aux holds the current figure and table numbers.
  2. Walk main.tex from \\begin{document} to \\appendix, following \\input, and collect
     every figure and table in reading order.
  3. Render figures (PDF -> PNG) and tables (LaTeX -> PDF -> PNG) into static/images/.
  4. Convert the captions and the abstract to HTML (math is left to KaTeX).
  5. Write index.html.

Usage:  python3 build_site.py [path/to/rest_arxiv_paper]
Needs pdflatex, bibtex, pdfcrop and pdftocairo on the PATH, and Pillow.
"""

import html
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
PAPER = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent / "rest_arxiv_paper"
IMAGES = HERE / "static" / "images"
PDFS = HERE / "static" / "pdfs"
MAX_WIDTH_PX = 2400
PAGE_URL = "https://fard-lab.github.io/REST/"
CODE_URL = "https://github.com/FARD-Lab/REST"
ARXIV_ID = ""  # set once the paper is on arXiv, e.g. "2610.01234"

TITLE = "Principled Thoughts for Latent Recursive LLM Systems"
AUTHORS = ["Fahd Seddik", "Fatemeh Fard"]
AFFILIATION = "FARD Lab, University of British Columbia"

LABEL_PREFIX = {"fig": "Figure", "tab": "Table", "sec": "Section", "app": "Appendix", "eq": "Eq.", "thm": "Theorem"}


def run(cmd, cwd):
    subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def compile_paper():
    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main"], PAPER)
    run(["bibtex", "main"], PAPER)
    for _ in range(2):
        run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main"], PAPER)


def read_labels():
    labels = {}
    for m in re.finditer(r"\\newlabel\{([^}@]+)\}\{\{([^}]*)\}\{([^}]*)\}", (PAPER / "main.aux").read_text()):
        labels[m.group(1)] = m.group(2)
    return labels


def braced(text, start):
    """Return (content, end) of the {...} group whose opening brace is at text[start]."""
    assert text[start] == "{"
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{" and text[i - 1] != "\\":
            depth += 1
        elif text[i] == "}" and text[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
    raise ValueError("unbalanced braces")


def command_arg(text, command):
    m = re.search(r"\\" + command + r"\*?(\[[^\]]*\])?\{", text)
    return braced(text, m.end() - 1)[0] if m else None


def body_floats():
    """Figures and tables of the main body, in reading order."""
    main = (PAPER / "main.tex").read_text()
    body = main.split("\\begin{document}", 1)[1].split("\\appendix", 1)[0]
    found = []

    def walk(text):
        for m in re.finditer(r"\\input\{([^}]+)\}", text):
            name = m.group(1)
            if re.match(r"(figures|tables|teaser)/", name):
                found.append(name)
            else:
                path = PAPER / (name if name.endswith(".tex") else name + ".tex")
                if path.exists():
                    walk(path.read_text())

    walk(body)
    return found


def latex_to_html(text, labels):
    """Convert caption-level LaTeX to HTML. Inline math stays as $...$ for KaTeX."""
    out, i = [], 0
    while i < len(text):
        if text[i] == "$":
            j = text.index("$", i + 1)
            out.append(html.escape(text[i:j + 1], quote=False))
            i = j + 1
            continue
        m = re.match(r"\\(textbf|emph|textit|texttt|cref|Cref|ref)\{", text[i:])
        if m:
            arg, end = braced(text, i + m.end() - 1)
            cmd = m.group(1)
            if cmd in {"cref", "Cref", "ref"}:
                parts = []
                for key in arg.split(","):
                    key = key.strip()
                    number = labels.get(key, "??")
                    prefix = LABEL_PREFIX.get(key.split(":")[0], "")
                    parts.append(number if cmd == "ref" else f"{prefix} {number}".strip())
                out.append(", ".join(parts))
            else:
                tag = {"textbf": "strong", "emph": "em", "textit": "em", "texttt": "code"}[cmd]
                out.append(f"<{tag}>{latex_to_html(arg, labels)}</{tag}>")
            i = end
            continue
        for src, dst in [("\\ ", " "), ("~", "&nbsp;"), ("\\%", "%"), ("\\&", "&amp;"), ("---", "\u2013"), ("--", "\u2013"), ("\\,", "\u2009")]:
            if text.startswith(src, i):
                out.append(dst)
                i += len(src)
                break
        else:
            if text[i] == "\\":
                m = re.match(r"\\[a-zA-Z]+\*?", text[i:])
                raise ValueError(f"unsupported LaTeX command in caption: {m.group(0) if m else text[i:i + 10]!r}")
            out.append(html.escape(text[i], quote=False))
            i += 1
    return re.sub(r"\s+", " ", "".join(out)).strip()


def png_from_pdf(pdf, png, dpi=300):
    with tempfile.TemporaryDirectory() as tmp:
        stem = Path(tmp) / "page"
        run(["pdftocairo", "-png", "-r", str(dpi), "-singlefile", str(pdf), str(stem)], tmp)
        im = Image.open(str(stem) + ".png").convert("RGB")
    if im.width > MAX_WIDTH_PX:
        im = im.resize((MAX_WIDTH_PX, round(im.height * MAX_WIDTH_PX / im.width)), Image.LANCZOS)
    im.save(png, optimize=True)
    return im.size


def render_tables(tables):
    """Typeset each table body with the paper's preamble, one per page, without its caption."""
    main = (PAPER / "main.tex").read_text()
    preamble = main.split("\\begin{document}", 1)[0]
    pages = []
    for name in tables:
        body = (PAPER / f"{name}.tex").read_text()
        body = re.sub(r"\\begin\{(wrap)?table\*?\}(\[[^\]]*\])?(\{[^}]*\})*", "", body)
        body = re.sub(r"\\end\{(wrap)?table\*?\}", "", body)
        pages.append("\\begin{center}\n" + body + "\n\\end{center}\n\\clearpage\n")
    doc = (
        preamble
        + "\\pagestyle{empty}\n\\begin{document}\n"
        + "\\renewcommand{\\caption}[2][]{}\\renewcommand{\\label}[1]{}\n"
        + "".join(pages)
        + "\\end{document}\n"
    )
    with tempfile.TemporaryDirectory(dir=PAPER) as tmp:
        tmp = Path(tmp)
        (tmp / "tables.tex").write_text(doc)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={tmp}", str(tmp / "tables.tex")],
                       cwd=PAPER, check=True, stdout=subprocess.DEVNULL)
        run(["pdfcrop", "--margins", "6", "tables.pdf", "cropped.pdf"], tmp)
        sizes = {}
        for i, name in enumerate(tables, start=1):
            one = tmp / f"t{i}.pdf"
            run(["pdfseparate", "-f", str(i), "-l", str(i), "cropped.pdf", one.name], tmp)
            sizes[name] = png_from_pdf(one, IMAGES / f"{Path(name).name}.png", dpi=360)
    return sizes


def figure_block(kind, number, caption, image, alt, width_pct):
    label = f"{kind} {number}."
    img = (f'<img src="static/images/{image}" alt="{html.escape(alt)}" loading="lazy" '
           f'style="width: {width_pct}%;">')
    if kind == "Table" and width_pct == 100:
        img = f'<div class="float-scroll">{img}</div>'
    cap = f'<figcaption><strong>{label}</strong> {caption}</figcaption>'
    inner = f"{cap}\n        {img}" if kind == "Table" else f"{img}\n        {cap}"
    return f'''      <figure class="paper-float paper-{kind.lower()}">
        {inner}
      </figure>'''


def plain(text):
    return re.sub(r"<[^>]+>", "", html.unescape(text))


def main():
    compile_paper()
    labels = read_labels()
    IMAGES.mkdir(parents=True, exist_ok=True)
    PDFS.mkdir(parents=True, exist_ok=True)
    for old in IMAGES.glob("*.png"):
        if old.name != "favicon.png":
            old.unlink()

    floats = body_floats()
    tables = [f for f in floats if f.startswith("tables/")]
    render_tables(tables)

    blocks = []
    for name in floats:
        tex = (PAPER / f"{name}.tex").read_text()
        caption_tex = command_arg(tex, "caption")
        label = command_arg(tex, "label")
        number = labels[label]
        caption = latex_to_html(caption_tex, labels)
        wrap = re.search(r"\\begin\{wrap(?:figure|table)\}(?:\[[^\]]*\])?\{[^}]*\}\{([\d.]+)\\linewidth\}", tex)
        width_pct = 100
        if wrap:
            width_pct = max(45, round(float(wrap.group(1)) * 100 * 1.4))
        if name.startswith("tables/"):
            blocks.append(figure_block("Table", number, caption, f"{Path(name).name}.png", plain(caption), width_pct))
        else:
            graphic = command_arg(tex, "includegraphics")
            pdf = PAPER / (graphic if graphic.endswith(".pdf") else graphic + ".pdf")
            png = f"{Path(name).name}.png"
            png_from_pdf(pdf, IMAGES / png)
            blocks.append(figure_block("Figure", number, caption, png, plain(caption), width_pct))

    abstract = latex_to_html((PAPER / "sections" / "abstract.tex").read_text(), labels)
    shutil.copy(PAPER / "main.pdf", PDFS / "REST.pdf")

    template = (HERE / "index.template.html").read_text()
    arxiv_url = f"https://arxiv.org/abs/{ARXIV_ID}" if ARXIV_ID else ""
    bibtex = (
        "@misc{seddik2026principled,\n"
        f"  title         = {{{TITLE}}},\n"
        f"  author        = {{{' and '.join(AUTHORS)}}},\n"
        "  year          = {2026},\n"
        + (f"  eprint        = {{{ARXIV_ID}}},\n  archivePrefix = {{arXiv}},\n  primaryClass  = {{cs.CL}},\n  url           = {{{arxiv_url}}}\n"
           if ARXIV_ID else f"  url           = {{{PAGE_URL}}}\n")
        + "}"
    )
    arxiv_button = (
        f'<a href="{arxiv_url}" target="_blank" rel="noopener" class="external-link button is-normal is-rounded is-dark">'
        '<span class="icon"><i class="ai ai-arxiv"></i></span><span>arXiv</span></a>'
        if ARXIV_ID else
        '<span class="button is-normal is-rounded is-dark is-static" aria-disabled="true">'
        '<span class="icon"><i class="ai ai-arxiv"></i></span><span>arXiv (soon)</span></span>'
    )
    page = (template
            .replace("{{TITLE}}", TITLE)
            .replace("{{AUTHORS_META}}", ", ".join(AUTHORS))
            .replace("{{AUTHORS}}", "\n              ".join(
                f'<span class="author-block">{a}{"," if i < len(AUTHORS) - 1 else ""}</span>' for i, a in enumerate(AUTHORS)))
            .replace("{{AFFILIATION}}", AFFILIATION)
            .replace("{{DESCRIPTION}}", html.escape(plain(abstract).split(". ")[0] + "."))
            .replace("{{ABSTRACT}}", abstract)
            .replace("{{ABSTRACT_PLAIN}}", html.escape(plain(abstract)))
            .replace("{{FLOATS}}", "\n".join(blocks))
            .replace("{{CODE_URL}}", CODE_URL)
            .replace("{{PAGE_URL}}", PAGE_URL)
            .replace("{{ARXIV_BUTTON}}", arxiv_button)
            .replace("{{BIBTEX}}", html.escape(bibtex)))
    leftover = re.findall(r"\{\{[A-Z_]+\}\}", page)
    if leftover:
        sys.exit(f"ERROR: unfilled placeholders {leftover}")
    if "\u2014" in page:
        sys.exit("ERROR: the page contains an em-dash")
    (HERE / "index.html").write_text(page)
    print(f"index.html: {len(blocks)} figures and tables")
    for name in floats:
        print("  ", name)


if __name__ == "__main__":
    main()
