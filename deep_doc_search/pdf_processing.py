import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF document."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    return text

# Module testing
if __name__ == "__main__":
    pdf_path = "data/test.pdf"  # Replace with an actual PDF file
    extracted_text = extract_text_from_pdf(pdf_path)
    print("Extracted text from PDF:")
    print(extracted_text[:500])  # Display the first 500 characters
