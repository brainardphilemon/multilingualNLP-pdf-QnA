#  Multilingual NLP PDF Q&A

Multilingual question-answer generator from a PDF

## Description

- It is a web-based application that allows users to upload PDFs in any language. The system extracts text using PDFPlumber and preprocesses it using re for better structure.Depending on the size of the PDF, it dynamically selects either the T5 model or Google Gemini AI to generate relevant questions.
- For answering the generated questions, the system leverages TF-IDF vectorization and cosine similarity to retrieve the most relevant text segments. Additionally, a translation layer enables users to interact in their preferred language, making the application accessible to a diverse audience.
## Getting Started

### Technologies Used

* Python

### Libraries Used

* Pandas
* PDFPlumber
* SKLearn

### LLMs Used

* Gemini



## Future
Build Frontend using React and Tailwind 
