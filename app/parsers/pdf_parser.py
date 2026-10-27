from pathlib import Path

import pymupdf
import json


def extract_pdf_pages(file_path):
    file_path = Path(file_path)
    pages = []

    with pymupdf.open(file_path) as document:
        if not document.is_pdf:
            raise ValueError("Choose a PDF file.")

        if document.needs_pass:
            raise ValueError("Password-protected PDFs are not supported yet.")

        for page_number, page in enumerate(document, start=1):
            pages.append({
                "file_name": file_path.name,
                "page_number": page_number,
                "text": page.get_text("text", sort=True).strip(),
            })

    return pages


if __name__ == "__main__":
    pages = extract_pdf_pages("storage/uploads/sample.pdf")

    output_path = Path("storage/extracted/sample.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(pages, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved {len(pages)} pages to {output_path}.")