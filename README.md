> Built by [Komal Agarwal](https://github.com/komalmagarwal) - 
> founder of Sola, exploring AI applications across healthcare and 
> financial services.

# AI for Financial Disclosures

This project demonstrates a Natural Language Processing (NLP) pipeline to extract, classify, and summarize financial disclosure documents.

## Why It Matters
Investors, analysts, and compliance teams often need to process large volumes of disclosure documents quickly.  
This project automates the process, saving time and improving accuracy.

## Tech Stack
- Python
- Hugging Face Transformers
- SpaCy
- Pandas

## Key Features
- Automatic classification of disclosure types
- Named Entity Recognition (NER) for companies, sectors, and financial terms
- Summarization of lengthy documents into concise briefs

## Sample Output
```json
{
  "document_type": "Annual Report",
  "entities": ["Goldman Sachs", "Q4 2024", "Revenue: $12.7B"],
  "summary": "Goldman Sachs Q4 2024 annual report highlights..."
}
```
