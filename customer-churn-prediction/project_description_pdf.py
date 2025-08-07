from fpdf import FPDF
import os
import re

# Paths to documentation and code files
README_PATH = os.path.join(os.path.dirname(__file__), 'README.md')
BUSINESS_REPORT_PATH = os.path.join(os.path.dirname(__file__), 'reports', 'business_insights_report.md')
REQUIREMENTS_PATH = os.path.join(os.path.dirname(__file__), 'requirements.txt')

# Helper to read file content
def read_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"[Could not read {path}: {e}]"

def clean_text_for_pdf(text):
    # Remove non-latin1 characters (e.g., emojis)
    return text.encode('latin-1', 'ignore').decode('latin-1')

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'Customer Churn Prediction Project - Step by Step Description', 0, 1, 'C')
        self.ln(2)
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 8, title, 0, 1, 'L')
        self.ln(1)
    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        for line in body.split('\n'):
            self.multi_cell(0, 5, clean_text_for_pdf(line))
        self.ln(2)

def main():
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # 1. Project Overview & Structure
    pdf.chapter_title('1. Project Overview & Structure')
    pdf.chapter_body(read_file(README_PATH))

    # 2. Business Insights & Recommendations
    pdf.chapter_title('2. Business Insights & Recommendations')
    pdf.chapter_body(read_file(BUSINESS_REPORT_PATH))

    # 3. Dependencies
    pdf.chapter_title('3. Dependencies')
    pdf.chapter_body(read_file(REQUIREMENTS_PATH))

    # Save PDF
    output_path = os.path.join(os.path.dirname(__file__), 'customer_churn_project_description.pdf')
    pdf.output(output_path)
    print(f"PDF generated: {output_path}")

if __name__ == '__main__':
    main()