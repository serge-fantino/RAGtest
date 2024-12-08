from llama_index.core import Settings
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import argparse
import yaml
import os
from ragtest.index_builder import load_or_create_index
from ragtest.query_engine import create_query_engine

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
            
            print("Sources utilisées (triées par pertinence):")
            if hasattr(response, 'source_nodes'):
                for i, source in enumerate(response.source_nodes):
                    print(f"\nSource {i+1} (score: {source.score:.3f}):")
                    print(f"Métadonnées: {source.node.metadata}")
                    print(f"Texte complet: {source.node.text}")
                    print("-" * 50)
                    
                print("\n" + "="*50)
                print("RÉPONSE FINALE :")
                print(response.response)
                print("="*50 + "\n")
            else:
                print("Aucune source pertinente trouvée")
                
        except Exception as e:
            print(f"\nErreur : {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='RAG System with local documents')
    parser.add_argument('doc_path', help='Path to the documents directory')
    parser.add_argument('--config', default='config.yml', help='Path to config file')
    parser.add_argument('--model-dir', help='Directory containing the models')
    parser.add_argument('--force-rebuild', action='store_true', 
                       help='Force rebuild the index instead of loading existing one')
    args = parser.parse_args()

    # Validation des chemins
    for path in [args.doc_path, args.config]:
        if not os.path.exists(path):
            print(f"Erreur: Le chemin {path} n'existe pas")
            return
    if args.model_dir and not os.path.exists(args.model_dir):
        print(f"Erreur: Le répertoire des modèles {args.model_dir} n'existe pas")
        return

    # Initialisation
    config = load_config(args.config)
    llm = init_llm(config, args.model_dir)
    embed_model = init_embedding_model(config)

    # Chargement ou création de l'index
    index = load_or_create_index(args.doc_path, llm, embed_model, config, args.force_rebuild)
    
    # Création du moteur de requête et lancement de la boucle
    query_engine = create_query_engine(index, llm, config)
    query_loop(query_engine)

if __name__ == "__main__":
    main()