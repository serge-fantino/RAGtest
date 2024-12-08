from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import os
import yaml
import re
from typing import List, Dict, Any
from pathlib import Path

def load_frontmatter_chunks(file_path: str) -> List[Document]:
    """
    Charge un fichier FrontMatter et retourne une liste de Documents.
    Le texte de chaque document est enrichi avec les métadonnées pour améliorer la recherche.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    chunks = content.split('\n---\n')
    documents = []
    
    # On ignore le premier élément s'il est vide
    if not chunks[0].strip():
        chunks = chunks[1:]
    
    # Traiter les chunks par paires
    for i in range(0, len(chunks)-1, 2):
        try:
            yaml_part = chunks[i].strip()
            content_part = chunks[i+1].strip() if i+1 < len(chunks) else ""
            
            # Parser les métadonnées YAML
            try:
                metadata = yaml.safe_load(yaml_part)
                if not metadata:
                    metadata = {'error': 'No metadata found'}
            except yaml.YAMLError as e:
                print(f"Warning: Erreur YAML dans le chunk {i//2} de {file_path}: {e}")
                continue
            
            # Modifier la construction du texte enrichi pour mettre plus en valeur les métadonnées
            enriched_text = ""
            if metadata:
                enriched_text += "MÉTADONNÉES IMPORTANTES:\n"
                if metadata.get('sprint'):
                    enriched_text += f"Ce document concerne le Sprint {metadata['sprint']}.\n"
                if metadata.get('date'):
                    enriched_text += f"Date de l'événement: {metadata['date']}.\n"
                if metadata.get('activity'):
                    enriched_text += f"Activité concernée: {metadata['activity']}.\n"
                if metadata.get('header_path'):
                    header_path = ' > '.join(metadata['header_path'])
                    enriched_text += f"Contexte hiérarchique: {header_path}.\n"
                
                enriched_text += "\nCONTENU DU DOCUMENT:\n"
            
            enriched_text += content_part
            
            # Aplatir les métadonnées pour Chroma
            flat_metadata = {
                key: str(value) if isinstance(value, (list, dict)) else value
                for key, value in metadata.items()
            }
            flat_metadata['source_file'] = file_path
            
            doc = Document(
                text=enriched_text,
                metadata=flat_metadata,
                excluded_embed_metadata_keys=['source_file'],  # Ne pas inclure le chemin du fichier dans l'embedding
                excluded_llm_metadata_keys=['source_file']     # Ne pas inclure le chemin du fichier dans le contexte LLM
            )
            documents.append(doc)
            print(f"✓ Document créé avec contexte enrichi: {' | '.join(context_parts)}")
            
        except Exception as e:
            print(f"Erreur lors du parsing du chunk {i//2} dans {file_path}: {str(e)}")
    
    if not documents:
        print(f"Warning: Aucun document valide trouvé dans {file_path}")
    else:
        print(f"✓ {len(documents)} documents valides extraits de {file_path}")
    
    return documents

def load_or_create_index(chunks_dir: str, llm: Any, embed_model: Any, force_rebuild: bool = False) -> VectorStoreIndex:
    """
    Charge ou crée un index à partir des chunks FrontMatter.
    """
    persist_dir = Path(chunks_dir).parent / "chroma_db"
    persist_dir.mkdir(exist_ok=True)
    print(f"Chemin de persistance : {persist_dir}")
    
    chroma_client = chromadb.PersistentClient(path=str(persist_dir))
    collection_name = Path(chunks_dir).name
    print(f"Nom de la collection : {collection_name}")
    
    if not force_rebuild:
        try:
            chroma_collection = chroma_client.get_collection(collection_name)
            print(f"\nCollection existante trouvée avec {chroma_collection.count()} documents.")
            
            # Afficher quelques exemples
            results = chroma_collection.get(limit=5)
            print("\nExemples de documents stockés:")
            for i, (doc_id, metadata, text) in enumerate(zip(
                results['ids'], results['metadatas'], results['documents']
            )):
                print(f"\nDocument {i+1}:")
                print(f"ID: {doc_id}")
                print(f"Métadonnées: {metadata}")
                print(f"Texte (premiers 200 caractères): {text[:200]}...")
            
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            return VectorStoreIndex.from_vector_store(
                vector_store,
                storage_context=storage_context,
                llm=llm,
                embed_model=embed_model
            )
        except Exception as e:
            print(f"\nErreur lors du chargement de l'index: {e}")
            print("Création d'un nouvel index...")
    
    # Création d'une nouvelle collection
    print("\nCréation d'une nouvelle collection...")
    if collection_name in [c.name for c in chroma_client.list_collections()]:
        chroma_client.delete_collection(collection_name)
    chroma_collection = chroma_client.create_collection(collection_name)
    
    # Chargement des documents
    md_files = [f for f in os.listdir(chunks_dir) if f.endswith('_chunks.md')]
    print(f"\nTraitement de {len(md_files)} fichiers...")
    
    all_documents = []
    for idx, file in enumerate(md_files, 1):
        file_path = os.path.join(chunks_dir, file)
        print(f"\nFichier {idx}/{len(md_files)}: {file}")
        documents = load_frontmatter_chunks(file_path)
        all_documents.extend(documents)
    
    # Création de l'index
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    print(f"\nCréation de l'index avec {len(all_documents)} documents...")
    return VectorStoreIndex.from_documents(
        all_documents,
        storage_context=storage_context,
        llm=llm,
        embed_model=embed_model,
        show_progress=True
    ) 