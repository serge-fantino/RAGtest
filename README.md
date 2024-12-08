# RAGtest - Personal Document RAG Experiment

by Serge Fantino

## Overview
This project is an experimental implementation of a Retrieval-Augmented Generation (RAG) system designed to work with highly personal documents. The goal is to create an AI assistant that can effectively understand and answer questions about personal content, such as daily journals, notes, or personal records.
This is also an experiment at augmented IDEA to learn how to better use Cursor and Claude - so you can expect most of the code and documentaiton to have been generated / augmented.

## Purpose
- Test the effectiveness of RAG systems with personal, contextual content
- Experiment with local LLM models for private data processing
- Create a foundation for building more sophisticated personal AI assistants
- Explore ways to make AI interactions more personalized and context-aware

## Current Focus
The project currently focuses on processing personal daily records and testing various approaches to:
- Document indexing and retrieval
- Context-aware responses
- Privacy-preserving local processing
- Optimization of response quality for personal content

## Status
This is an evolving experimental project. The implementation details and architecture may change frequently as new approaches are tested and refined.

## Installation and Usage Guide

### 1. Initial Setup

#### Clone the project

```bash
git clone https://github.com/your-repo/RAGtest.git
cd RAGtest
```

#### Install with Poetry

```bash
# Install Poetry if not already done
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
LLAMA_METAL=1 poetry install  # On MacOS for Metal support

# OR
poetry install  # On other systems
```

### 2. Data and Models Setup

#### Directory Structure

Example of the directory structure (but you can host your documetns and models somewhere else):

```
RAGtest/
├── data/                      # Your documents to analyze
│   └── my_documents/        
│       └── *.html            # Your files (HTML, TXT, etc.)
│   └── chroma_db/             # Embeddings database directory (created automatically)
├── models/                    # LLM models directory
│   └── dolphin-2.8-mistral-7b-v02-GGUF/ # it is the default model provided in config.yaml
│       └── dolphin-2.8-mistral-7b-v02-Q4_0.gguf
└── ragtest/
    ├── config.yaml           # Configuration
    └── queryDocs.py          # Main script
```

#### Configuration

You can adapt the provided config.yaml file to your needs - it expects the model to be in the models directory and configured for dolphin-2.8-mistral-7b-v02-GGUF


### 3. Running the Test

#### Activate Poetry Environment

```bash
poetry shell
```

#### Run the Script

```bash
# Command structure
python ragtest/queryDocs.py [documents_path] --model-dir [models_path] --config [config_path]

# Example
python ragtest/queryDocs.py ./data/my_documents --model-dir ./models --config ./ragtest/config.yaml
```

### 4. Usage

Once launched, the script:
1. Loads or creates an embeddings database for your documents, the database is stored in the next to the documents directory
2. Starts an interactive interface where you can ask questions

Example questions:

```
Your question: what happened on January 15th?
Your question: summarize last week's events
Your question: q    # To quit
```

### 5. Important Notes

This is a educational project, don't expect the answers to be meaningful - see my comments in the History chapters. Anyway if you find it useful, or succeed to make it work with your own documents, let me know!

1. **LLM Models**: 
   - You need to download GGUF models yourself
   - Place them in the `models/` directory following the expected structure

2. **Data**:
   - The system accepts various document formats
   - An embeddings database will be created on first run
   - Embeddings are saved for subsequent uses

3. **Performance**:
   - On MacOS, ensure Metal support is activated
   - Currently fine tuned for a Mac Mini M4PRO maximum configuration (:D) 
   - Adjust `n_threads` according to your CPU
   - Modify `similarity_top_k` for more or less context

4. **Troubleshooting**:
   - Memory errors: reduce `n_ctx` and `context_window`
   - Off-topic answers: increase `similarity_threshold`
   - No answers: decrease `similarity_threshold`

## Privacy Note
All processing is done locally to ensure personal data privacy. The system uses local LLM models and local storage for vector embeddings.

## History

### 2024-12-06

#### What?
- First version of the project
- Basic RAG system with local LLM and ChromaDB for vector storage
- Simple query interface for testing
- Initial testing with a small dataset of personal documents

#### Feedback

##### Building the first version

- I've been using Cusor & Claude Sonnet to build it
- Claude is very helpful but I found difficult to get the import / installation of all packages right (lot's of import changes and Claude mixed different version)
- Also Claude generates wrong code the first time for reloading the embedings from ChromaDB - took me too long to identify the problem (and Claude was completely blind either), so it is best practice to **read the code !!!** - but finaly once I identified the issue Claude was able to fix it alone.
- Must explicitly active metal optimisation to leverage the GPU : this is huge hit for performance

#### How it perform?

- overall evaluation is fast. Also I added persistence of the database so once the database is initialized we can start querying with minimal delay
- with metal optimisation the dolphin model perform pretty fast
- but to be honnest the results are not very good:
    - looks like the model only retrieves 2 documents at a time, which is not enought
    - most of the time complexe queries returns no results or completely out of scope
    - the only questions that gets something real: what I was doing the 11th November? you were OFF (yeah !)

So for now it is a complete failure :) 

#### What's next?

- use different model / embedding model
- preprocessing the data to analyse ? this is a raw html export and the layout of the document is not intended for this... should generate something more suitable for learning
- see [Construire son RAG (Retrieval Augmented Generation) avec LlamaIndex et le modèle LLM Vigogne](https://zonetuto.fr/intelligence-artificielle/construire-rag-llamaindex-llm-vigogne/)
- hybrid search ? cf. [Optimiser les performances de RAG : Approches avancées pour une application efficace de la recherche d’informations avec des modèles LLM](https://mazen-alsarem.com/optimiser-les-performances-de-rag-approches-avancees-pour-une-application-efficace-de-la-recherche-dinformations-avec-des-modeles-llm/)

### 2024-12-07

#### What?

- refactoring of initial code to structure components and allow running on a document directory / automatic management of embeddings database creation
- isolation of model configuration and query engine settings in a configuration file

Ready to share on Github!

Also quick test with another model: Mistral-Nemo-Instruct-2407-Q4_K_M.gguf
- results are even worse and much slower (makes sense)
- find a more recent 4-bit model?

#### Also later on Saturday:

- I tried to implement a new indexing strategy that leverages the structured hierarchy of the documents (sprint, date, activity), in order to enrich the chunks with metadata
- And I was expecting this to allow better retrieval of relevant documents, based on the query...
- But to be honnest this is not working at all...
- There are hardly no correlation between the qeury referencing key elements (e.g. date, sprint number...) and teh retrieval of the source documents
- Thus the reply are completely out of scope / context.
- I also tried to use the LLM to first preprocess the user query in order to clearly identify the metadata, and use that to enforce the search in the vector database... but the results are quite unpredictable - I see a lot of hallucination and the results are not better even after adding strong rules in prompt to avoid it.

I will push this work in a working branch, because this is clearly not good at all.
- So this is a dead end for now...
- I am not even convinced that tehre is a way to do better : I am just questionning the quality of my data sample, maybe it is too fuzy for this application.
- I should consider using a different use case with more coherent data


