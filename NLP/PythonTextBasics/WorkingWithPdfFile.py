import PyPDF2

f = open('US_Declaration.pdf', 'rb')
pdf_reader = PyPDF2.PdfFileReader(f)
print(pdf_reader.numPages)
page_one = pdf_reader.getPage(0)
page_two = pdf_reader.getPage(1)
page_one_text = page_one.extractText()
print(page_one_text)

# We can not write to PDFs using Python because of the differences between the single string type of Python,
# and the variety of fonts, placements, and other parameters that a PDF could have. What we can do is copy pages and
# append pages to the end.

pdf_writer = PyPDF2.PdfFileWriter()
pdf_writer.addPage(page_two)
pdf_output = open("Some_New_Doc.pdf","wb")
pdf_writer.write(pdf_output)

pdf_output.close()
f.close()

f = open('US_Declaration.pdf', 'rb')
# List of every page's text.
# The index will correspond to the page number.
pdf_text = [0]  # zero is a placehoder to make page 1 = index 1
pdf_reader = PyPDF2.PdfFileReader(f)
for p in range(pdf_reader.numPages):
    page = pdf_reader.getPage(p)
    pdf_text.append(page.extractText())
f.close()
print(pdf_text)
print(pdf_text[2])

for page in pdf_text:
    print(page)
    print("\n")
    print("\n")

