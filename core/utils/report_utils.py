import os

import markdown
from xhtml2pdf import pisa

from ..exceptions import FileOperationException


def save_report_to_disk(
    report_content: str,
    filename: str,
    reports_dir: str = "reports",
) -> tuple[str, str]:
    os.makedirs(reports_dir, exist_ok=True)

    markdown_path = os.path.join(reports_dir, f"{filename}.md")
    pdf_path = os.path.join(reports_dir, f"{filename}.pdf")

    try:
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(report_content)
    except Exception as e:
        raise FileOperationException(f"Failed to save markdown: {str(e)}") from e

    try:
        html_content = markdown.markdown(
            report_content, extensions=["extra", "codehilite", "tables", "toc"]
        )
        styled_html = _create_styled_html(html_content)

        with open(pdf_path, "wb") as pdf_file:
            pisa_status = pisa.CreatePDF(styled_html, dest=pdf_file)

        if pisa_status.err:
            raise FileOperationException("PDF generation had errors")

    except Exception as e:
        raise FileOperationException(f"Failed to generate PDF: {str(e)}") from e

    return markdown_path, pdf_path


def _create_styled_html(html_content: str) -> str:
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            @page {{
                size: A4;
                margin: 2cm;
            }}
            body {{
                font-family: 'Helvetica', 'Arial', sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 100%;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                margin-top: 30px;
            }}
            h2 {{
                color: #34495e;
                border-bottom: 2px solid #95a5a6;
                padding-bottom: 8px;
                margin-top: 25px;
            }}
            h3 {{
                color: #34495e;
                margin-top: 20px;
            }}
            a {{
                color: #3498db;
                text-decoration: none;
            }}
            a:hover {{
                text-decoration: underline;
            }}
            p {{
                margin: 12px 0;
                text-align: justify;
            }}
            ul, ol {{
                margin: 12px 0;
                padding-left: 30px;
            }}
            li {{
                margin: 8px 0;
            }}
            code {{
                background-color: #f4f4f4;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
            }}
            blockquote {{
                border-left: 4px solid #3498db;
                padding-left: 20px;
                margin: 20px 0;
                color: #555;
                font-style: italic;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
