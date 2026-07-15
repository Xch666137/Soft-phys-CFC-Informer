import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

from pypdf import PdfReader


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_screen(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| AE"):
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) < 7:
            continue
        rows.append(
            {
                "id": parts[0],
                "bucket": parts[1],
                "year": parts[3],
                "cites": parts[4],
                "doi": parts[5].strip("`"),
                "title": parts[6],
            }
        )
    return rows


def best_match(file_name, rows):
    stem = normalize(Path(file_name).stem)
    best = None
    best_score = -1
    for row in rows:
        score = SequenceMatcher(None, stem, normalize(row["title"])).ratio()
        if score > best_score:
            best = row
            best_score = score
    return best, best_score


def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_between(text, start_patterns, end_patterns, max_chars=4000):
    lower = text.lower()
    starts = []
    for pattern in start_patterns:
        match = re.search(pattern, lower, re.I)
        if match:
            starts.append(match.end())
    if not starts:
        return ""
    start = min(starts)
    end = len(text)
    for pattern in end_patterns:
        match = re.search(pattern, lower[start:], re.I)
        if match:
            end = min(end, start + match.start())
    return text[start:end].strip()[:max_chars]


def sentence_hits(text, terms, limit=20):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    hits = []
    for sentence in sentences:
        low = sentence.lower()
        if any(term in low for term in terms):
            s = sentence.strip()
            if 60 <= len(s) <= 450:
                hits.append(s)
        if len(hits) >= limit:
            break
    return hits


def section_headings(text, limit=40):
    headings = []
    patterns = [
        r"(?:^|\s)(\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z0-9,\-/() ]{4,80})",
        r"(?:^|\s)(Abstract|Introduction|Methodology|Methods|Case study|Results|Discussion|Conclusion|Conclusions|Nomenclature|Appendix)\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            heading = re.sub(r"\s+", " ", match.group(1)).strip()
            if heading not in headings:
                headings.append(heading)
            if len(headings) >= limit:
                return headings
    return headings


def read_pdf(path):
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    raw = "\n".join(pages)
    text = clean_text(raw)
    return reader, text


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: extract_applied_energy_pdfs.py <pdf_dir> <screen_md> <out_dir>")
    pdf_dir = Path(sys.argv[1])
    screen_md = Path(sys.argv[2])
    out_dir = Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [r for r in parse_screen(screen_md) if r["id"] <= "AE15"]
    outputs = []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        match, score = best_match(pdf.name, rows)
        reader, text = read_pdf(pdf)
        first_pages = text[:25000]
        abstract = extract_between(
            first_pages,
            [r"\babstract\b"],
            [r"\bkeywords?\b", r"\b1\.?\s+introduction\b", r"\bintroduction\b"],
            max_chars=3000,
        )
        conclusion = extract_between(
            text,
            [r"\bconclusions?\b"],
            [r"\bdeclaration of competing interest\b", r"\bcredit authorship\b", r"\backnowledg", r"\breferences\b"],
            max_chars=3500,
        )
        keywords = extract_between(
            first_pages,
            [r"\bkeywords?\b"],
            [r"\b1\.?\s+introduction\b", r"\bintroduction\b"],
            max_chars=1000,
        )
        outputs.append(
            {
                "id": match["id"] if match else None,
                "match_score": round(score, 3),
                "bucket": match["bucket"] if match else None,
                "doi": match["doi"] if match else None,
                "title": match["title"] if match else pdf.stem,
                "file_name": pdf.name,
                "pages": len(reader.pages),
                "char_count": len(text),
                "chars_per_page": round(len(text) / max(len(reader.pages), 1), 1),
                "text_quality": "good" if len(text) / max(len(reader.pages), 1) > 800 else "check",
                "abstract": abstract,
                "keywords": keywords,
                "headings": section_headings(text),
                "baseline_metric_hits": sentence_hits(
                    text,
                    [
                        "baseline",
                        "benchmark",
                        "compared with",
                        "comparison",
                        "mae",
                        "rmse",
                        "mape",
                        "crps",
                        "pinball",
                        "ablation",
                    ],
                    limit=16,
                ),
                "operational_hits": sentence_hits(
                    text,
                    [
                        "dispatch",
                        "market",
                        "ramping",
                        "reserve",
                        "scheduling",
                        "operation",
                        "economic",
                        "cost",
                        "uncertainty",
                        "risk",
                    ],
                    limit=16,
                ),
                "conclusion": conclusion,
            }
        )

    json_path = out_dir / "applied_energy_extracted_sections.json"
    json_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Applied Energy PDF Extraction Summary", ""]
    for item in sorted(outputs, key=lambda x: x["id"] or ""):
        lines.extend(
            [
                f"## {item['id']} {item['title']}",
                "",
                f"- file: `{item['file_name']}`",
                f"- doi: `{item['doi']}`",
                f"- pages: {item['pages']}, chars: {item['char_count']}, quality: {item['text_quality']}",
                f"- bucket: {item['bucket']}",
                "",
                "### Abstract",
                "",
                item["abstract"][:1800] if item["abstract"] else "[not found]",
                "",
                "### Baseline / Metric Hits",
                "",
            ]
        )
        lines.extend([f"- {hit}" for hit in item["baseline_metric_hits"][:8]] or ["- [none found]"])
        lines.extend(["", "### Operational Hits", ""])
        lines.extend([f"- {hit}" for hit in item["operational_hits"][:8]] or ["- [none found]"])
        lines.extend(["", "### Conclusion", "", item["conclusion"][:1800] if item["conclusion"] else "[not found]", ""])
    md_path = out_dir / "applied_energy_extraction_summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
