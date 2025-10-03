import PyPDF2
import os

PDF_FOLDER = "../docs/"
TEXT_OUTPUT = "../embeddings/all_laws.txt"

all_text = ""

for filename in os.listdir(PDF_FOLDER):
    if filename.endswith(".pdf"):
        with open(os.path.join(PDF_FOLDER, filename), 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                all_text += page.extract_text() + "\n"

with open(TEXT_OUTPUT, "w", encoding="utf-8") as f:
    f.write(all_text)

print("Text extracted successfully!")
