"""Render the manuscript + SI markdown to .docx in the M2M house format:
Times New Roman, all 12 pt, simple bold section titles. Markdown tables -> Word tables,
**bold**/_italic_ inline, blockquotes (prompts/captions) styled, and figures embedded at
their captions. Run with system python3 (has python-docx). Usage: python3 build_docx.py
"""
import re, sys
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.oxml.ns import qn

ROOT = Path(__file__).parent.parent
FIGDIR = ROOT / "figures"
FIG_MAP = {  # caption-key -> png
    "Fig. 1": "fig1_design.png",
    "Fig. 2": "fig2_content_diversity.png",
    "Fig. 3": "fig3_role_transitions.png",
    "Fig. 4": "fig4_planted_conclusion.png",
    "Fig. 5": "fig5_fanin_topology.png",
    "Fig. 6": "fig6_two_failure_modes.png",
    "Fig. S1": "figS1_readout_validation.png",
}


def set_font(doc):
    st = doc.styles["Normal"].font
    st.name = "Times New Roman"; st.size = Pt(12)
    rpr = doc.styles["Normal"].element.get_or_add_rPr().get_or_add_rFonts()
    for a in ("w:ascii", "w:hAnsi", "w:cs"):
        rpr.set(qn(a), "Times New Roman")


def add_runs(p, text):
    """Parse `code`, **bold**, and _italic_ / *italic* into runs.
    Backtick code spans are matched first and rendered monospace, so underscores
    inside code labels (e.g. evidence_only) are preserved and never read as italics."""
    for tok in re.split(r"(`[^`]+`|\*\*.+?\*\*|_[^_]+_|\*[^*]+\*)", text):
        if not tok:
            continue
        if tok.startswith("`") and tok.endswith("`"):
            r = p.add_run(tok[1:-1]); r.font.name = "Courier New"
            r._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:cs"), "Courier New")
        elif tok.startswith("**") and tok.endswith("**"):
            r = p.add_run(tok[2:-2]); r.bold = True
        elif (tok.startswith("_") and tok.endswith("_")) or (tok.startswith("*") and tok.endswith("*")):
            r = p.add_run(tok[1:-1]); r.italic = True
        else:
            p.add_run(tok)


def heading(doc, text, size=12):
    p = doc.add_paragraph(); p.space_after = Pt(4)
    r = p.add_run(text); r.bold = True; r.font.size = Pt(size)
    return p


def add_table(doc, rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [c for c in cells if not all(re.fullmatch(r":?-+:?", x or "-") for x in c)]  # drop separator
    if not cells:
        return
    t = doc.add_table(rows=len(cells), cols=len(cells[0])); t.style = "Table Grid"
    for i, row in enumerate(cells):
        for j, val in enumerate(row):
            if j >= len(t.rows[i].cells):
                continue
            cell = t.rows[i].cells[j]; cell.paragraphs[0].text = ""
            p = cell.paragraphs[0]
            # bold header row, parse inline
            if i == 0:
                r = p.add_run(re.sub(r"\*\*", "", val)); r.bold = True
            else:
                add_runs(p, val)
            for rn in p.runs:
                rn.font.name = "Times New Roman"; rn.font.size = Pt(11 if i else 11)
    doc.add_paragraph()


def render_into(doc, md_path):
    lines = Path(md_path).read_text().splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i].rstrip()
        if not ln.strip():
            i += 1; continue
        # table block
        if ln.lstrip().startswith("|") and i+1 < len(lines) and re.search(r"\|\s*:?-+", lines[i+1]):
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i]); i += 1
            add_table(doc, block); continue
        # SI/main blockquote table ("> | ... |") — strip "> " then treat as table
        if ln.lstrip().startswith("> |") and i+1 < len(lines) and re.search(r"\|\s*:?-+", lines[i+1]):
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("> |"):
                block.append(lines[i].lstrip()[2:]); i += 1
            add_table(doc, block); continue
        # headings
        m = re.match(r"^(#{1,4})\s+(.*)", ln)
        if m:
            heading(doc, m.group(2)); i += 1; continue
        # horizontal rule
        if re.fullmatch(r"-{3,}", ln.strip()):
            i += 1; continue
        # blockquote (figure caption or prompt)
        if ln.lstrip().startswith(">"):
            text = ln.lstrip()[1:].strip()
            figkey = next((k for k in FIG_MAP if text.startswith(f"**{k}")), None)
            if figkey and (FIGDIR / FIG_MAP[figkey]).exists():
                pic = doc.add_paragraph(); pic.alignment = 1
                run = pic.add_run(); run.add_picture(str(FIGDIR / FIG_MAP[figkey]), width=Inches(6.2))
            p = doc.add_paragraph(); p.paragraph_format.left_indent = Inches(0.3)
            add_runs(p, text)
            for rn in p.runs:
                rn.font.size = Pt(11)
            i += 1; continue
        # bullet
        if ln.lstrip().startswith("- "):
            p = doc.add_paragraph(style="List Bullet"); add_runs(p, ln.lstrip()[2:]); i += 1; continue
        # normal paragraph
        p = doc.add_paragraph(); add_runs(p, ln); i += 1


def render(md_path, docx_path):
    doc = Document(); set_font(doc)
    render_into(doc, md_path)
    doc.save(docx_path)
    print(f"wrote {docx_path}")


def render_merged(md_paths, docx_path):
    """Render several markdown sources into one .docx, page-broken between sources.
    Figures and tables are embedded inline at their captions/positions."""
    from docx.enum.text import WD_BREAK
    doc = Document(); set_font(doc)
    for idx, md in enumerate(md_paths):
        if idx:
            doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
        render_into(doc, md)
    doc.save(docx_path)
    print(f"wrote {docx_path}")


if __name__ == "__main__":
    # Submission files (PNAS Nexus uploads the main manuscript and the SI as SEPARATE files):
    render(ROOT / "Manuscript Draft v5.md", ROOT / "M2M_Manuscript.docx")
    render(ROOT / "Supplementary Materials.md", ROOT / "M2M_Supplementary_Information.docx")
    # Merged reading/circulation copy (NOT a submission artifact):
    render_merged(
        [ROOT / "Manuscript Draft v5.md", ROOT / "Supplementary Materials.md"],
        ROOT / "M2M_Collective_Belief_Dynamics_Manuscript_with_SI.docx",
    )
