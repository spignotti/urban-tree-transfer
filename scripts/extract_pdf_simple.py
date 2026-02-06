#!/usr/bin/env python3
"""Simple PDF text extractor."""

import sys
from pathlib import Path

# Try different PDF libraries
pdf_lib = None
try:
    import pypdf

    PdfReader = pypdf.PdfReader
    pdf_lib = "pypdf"
except ImportError:
    try:
        from PyPDF2 import PdfReader

        pdf_lib = "PyPDF2"
    except ImportError:
        try:
            import pdfplumber

            pdf_lib = "pdfplumber"
        except ImportError:
            pass

if not pdf_lib:
    print("No PDF library available!")
    print("Install with: uv add --dev pypdf")
    sys.exit(1)

# Find PDFs
temp_dir = Path(
    "/Users/silas/Documents/projects/uni/Geo Projektarbeit/urban-tree-transfer/docs/literature/temp"
)
output_file = Path("/tmp/pdf_extracts.txt")

with open(output_file, "w", encoding="utf-8") as out:
    out.write(f"Using PDF library: {pdf_lib}\n")
    out.write("=" * 80 + "\n\n")

    for pdf_file in sorted(temp_dir.glob("*.pdf")):
        out.write(f"\n{'=' * 80}\n")
        out.write(f"FILE: {pdf_file.name}\n")
        out.write(f"{'=' * 80}\n\n")

        try:
            if pdf_lib == "pdfplumber":
                import pdfplumber

                with pdfplumber.open(pdf_file) as pdf:
                    for _i, page in enumerate(pdf.pages[:3]):
                        text = page.extract_text()
                        if text:
                            out.write(text[:2000])
                            out.write("\n")
            else:
                reader = PdfReader(str(pdf_file))
                for _i in range(min(3, len(reader.pages))):
                    text = reader.pages[_i].extract_text()
                    if text:
                        out.write(text[:2000])
                        out.write("\n")
        except Exception as e:
            out.write(f"ERROR: {e}\n")

        out.write("\n[...next file...]\n\n")

print(f"Output written to: {output_file}")
