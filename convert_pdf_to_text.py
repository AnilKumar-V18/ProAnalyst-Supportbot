from pypdf import PdfReader
import os

# Convert PDF to text
def convert_pdf_to_text(pdf_path, output_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    
    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        full_text += f"--- Page {page_num} ---\n{text}\n\n"
        print(f"Extracted page {page_num}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    print(f"\n✅ Converted {len(reader.pages)} pages")
    print(f"Total characters: {len(full_text)}")
    print(f"Saved to: {output_path}")

# Run conversion - NOTE the filename with spaces
pdf_file = "API Documentation Partial.pdf"  # ← Fixed: with spaces
txt_file = "upwork_api_reference.txt"

if os.path.exists(pdf_file):
    print(f"Found file: {pdf_file}")
    convert_pdf_to_text(pdf_file, txt_file)
else:
    print(f"❌ Error: {pdf_file} not found")
    print("Files in current directory:")
    for file in os.listdir('.'):
        if file.endswith('.pdf'):
            print(f"  - {file}")