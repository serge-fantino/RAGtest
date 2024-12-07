from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
import os
import argparse
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def init_llm(config, model_dir=None):
    llm_config = config['llm']
    # Construit le chemin complet du modèle
    if model_dir:
        actual_model_path = os.path.join(model_dir, llm_config['model_name'])
    else:
        actual_model_path = llm_config['model_path']  # fallback sur le chemin complet
    
    print(f"Chargement du modèle LLM: {actual_model_path}")
    return LlamaCPP(
        model_path=actual_model_path,
        model_kwargs=llm_config['model_kwargs'],
        temperature=llm_config.get('temperature', 0.1),
        max_new_tokens=llm_config.get('max_new_tokens', 512),
        context_window=llm_config.get('context_window', 3900),
        generate_kwargs=llm_config.get('generate_kwargs', {}),
        verbose=llm_config.get('verbose', True),
    )

def init_embedding_model(config):
    embed_config = config['embedding']
    return HuggingFaceEmbedding(model_name=embed_config['model_name'])

def load_or_create_index(doc_path, llm, embed_model):
    PERSIST_DIR = os.path.join(os.path.dirname(doc_path), "chroma_db")
    os.makedirs(PERSIST_DIR, exist_ok=True)
    print(f"Chemin de persistance : {PERSIST_DIR}")
    
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection_name = os.path.basename(os.path.dirname(doc_path))
    print(f"Nom de la collection : {collection_name}")
    try:
        chroma_collection = chroma_client.get_collection(collection_name)
        print("Collection existante chargée.")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
            llm=llm,
            embed_model=embed_model
        )
        print(f"Index reconstruit. Collection size: {chroma_collection.count()}")
    except:
        print("Création d'une nouvelle collection...")
        chroma_collection = chroma_client.create_collection(collection_name)
        documents = SimpleDirectoryReader(doc_path).load_data()
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            llm=llm,
            embed_model=embed_model
        )
        print(f"Documents indexés. Collection size: {chroma_collection.count()}")
    
    return index

def create_query_engine(index, llm, config):
    query_config = config['query_engine']
    return index.as_query_engine(
        llm=llm,
        similarity_top_k=query_config.get('similarity_top_k', 3),
        similarity_threshold=query_config.get('similarity_threshold', 0.7),
        response_mode=query_config.get('response_mode', "tree_summarize"),
        structured_answer_filtering=query_config.get('structured_answer_filtering', True),
        verbose=query_config.get('verbose', True),
        text_qa_template=query_config.get('template', """
        Tu es un assistant précis et factuel. Utilise uniquement les informations du CONTEXTE ci-dessous pour répondre à la QUESTION.
        Si tu ne trouves pas l'information dans le contexte, dis-le clairement.
        Ne fais pas de suppositions et reste objectif.

        CONTEXTE:
        {context_str}

        QUESTION:
        {query_str}

        RÉPONSE FACTUELLE:
        """),
    )

def query_loop(query_engine):
    while True:
        question = input("\nVotre question : ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nAu revoir !")
            break
        
        if not question:
            print("Veuillez poser une question...")
            continue
        
        try:
            print("\nRecherche en cours...\n")
            response = query_engine.query(question)
            
            print("Sources utilisées :")
            if hasattr(response, 'source_nodes'):
                for i, source in enumerate(response.source_nodes):
                    print(f"\nSource {i+1} (score: {source.score:.3f}):")
                    print(f"Extrait : {source.node.text[:200]}...")
                    
                print("\n" + "="*50)
                print("RÉPONSE FINALE :")
                print(response.response)
                print("="*50 + "\n")
            else:
                print("Aucune source pertinente trouvée")
                
        except Exception as e:
            print(f"\nErreur : {str(e)}")
            if "context window" in str(e):
                print("Le contexte est trop grand, essayons avec moins de sources...")
                try:
                    response = query_engine.query(question, similarity_top_k=2)
                    print(f"\nRéponse alternative : {response.response}\n")
                except Exception as e2:
                    print(f"Nouvelle erreur : {str(e2)}")

def main():
    parser = argparse.ArgumentParser(description='RAG System with local documents')
    parser.add_argument('doc_path', help='Path to the documents directory')
    parser.add_argument('--config', default='config.yml', help='Path to config file')
    parser.add_argument('--model-dir', help='Directory containing the models')
    args = parser.parse_args()

    if not os.path.exists(args.doc_path):
        print(f"Erreur: Le chemin {args.doc_path} n'existe pas")
        return

    if not os.path.exists(args.config):
        print(f"Erreur: Le fichier de configuration {args.config} n'existe pas")
        return

    if args.model_dir and not os.path.exists(args.model_dir):
        print(f"Erreur: Le répertoire des modèles {args.model_dir} n'existe pas")
        return

    print(f"Utilisation du répertoire: {os.path.abspath(args.doc_path)}")
    if args.model_dir:
        print(f"Répertoire des modèles: {os.path.abspath(args.model_dir)}")
    
    config = load_config(args.config)
    llm = init_llm(config, args.model_dir)
    embed_model = init_embedding_model(config)
    index = load_or_create_index(args.doc_path, llm, embed_model)
    query_engine = create_query_engine(index, llm, config)
    query_loop(query_engine)

if __name__ == "__main__":
    main()