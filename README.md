# Re-ranking with Query Expansion

## Overview

This project consists of three main Python files and utility modules for document retrieval, re-ranking, and language model integration. The purpose of this project is to showcase the integration of language models, document retrieval, and re-ranking techniques for enhanced information retrieval.

## Files

1. **`main.py`**
   - The main script to execute the entire workflow. It loads data from a PDF file, performs document retrieval, re-ranks the documents using a Cross Encoder, and incorporates automatic query expansion.

2. **`reranker.py`**
   - Contains functions related to re-ranking retrieved documents using a Cross Encoder.

3. **`query_expansion.py`**
   - Includes functions for generating question responses based on input text, using a language model and a predefined prompt.

4. **`helper_utils.py`**
   - Utility functions for reading PDF files, chunking texts, loading data into a Chroma database, word wrapping, and projecting embeddings.

5. **`model.py`**
   - Defines a Watson Machine Learning model using the IBM Watson platform and integrates it into a LangChain-based language model.

## How to Run

1. Ensure you have the necessary dependencies installed. You can install them using:

   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   - Ensure you have the required environment variables set, such as `GA_GENAI_URL`, `GA_GENAI_KEY`, and `GA_PROJECT_ID`.

3. Execute the main script:

   ```bash
   python main.py
   ```

## Dependencies

- `numpy`
- `tqdm`
- `chromadb`
- `sentence_transformers`
- `langchain`
- `ibm_watson_machine_learning`

## Credits

- This project was created by Charan H U.

Feel free to customize this README to provide more details about the project structure, usage, and any additional instructions.
