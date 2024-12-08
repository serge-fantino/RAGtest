from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import os
import yaml
from typing import List, Dict, Any
from pathlib import Path
from collections import defaultdict

def load_frontmatter_chunks(file_path: str, required_metadata: List[str]) -> List[Document]:
    """
    Charge un fichier FrontMatter et regroupe les chunks par métadonnées communes.
    Ne garde que les documents ayant toutes les métadonnées requises.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    chunks = content.split('\n---\n')
    if not chunks[0].strip():
        chunks = chunks[1:]
    
    # Dictionnaire pour regrouper les contenus par métadonnées
    grouped_chunks = defaultdict(list)
    
    # Première passe : regrouper les chunks par métadonnées
    for i in range(0, len(chunks)-1, 2):
        try:
            yaml_part = chunks[i].strip()
            content_part = chunks[i+1].strip() if i+1 < len(chunks) else ""
            
            try:
                metadata = yaml.safe_load(yaml_part)
                if not metadata:
                    metadata = {'error': 'No metadata found'}
                
                # Convertir les métadonnées en une clé de regroupement
                meta_key = tuple(sorted([
                    (k, str(v)) for k, v in metadata.items()
                    if k in ['sprint', 'date', 'activity', 'header_path']
                ]))
                
                # Regrouper le contenu
                grouped_chunks[meta_key].append({
                    'content': content_part,
                    'metadata': metadata
                })
                
            except yaml.YAMLError as e:
                print(f"Warning: Erreur YAML dans le chunk {i//2} de {file_path}: {e}")
                continue
                
        except Exception as e:
            print(f"Erreur lors du parsing du chunk {i//2} dans {file_path}: {str(e)}")
    
    # Deuxième passe : créer les documents regroupés
    documents = []
    skipped_chunks = 0
    for meta_key, chunks_group in grouped_chunks.items():
        try:
            # Récupérer les métadonnées du premier chunk du groupe
            metadata = chunks_group[0]['metadata']
            
            # Vérifier les métadonnées requises
            missing_metadata = [
                field for field in required_metadata 
                if field not in metadata or not metadata[field]
            ]
            
            if missing_metadata:
                print(f"⚠ Skip document: métadonnées manquantes {missing_metadata}")
                skipped_chunks += len(chunks_group)
                continue
            
            # S'assurer que le sprint est une string
            if 'sprint' in metadata:
                metadata['sprint'] = str(metadata['sprint'])
            
            # Construire le texte enrichi pour le groupe
            enriched_text = "MÉTADONNÉES IMPORTANTES:\n"
            if metadata.get('sprint'):
                enriched_text += f"Ce document concerne le Sprint {metadata['sprint']}.\n"
            if metadata.get('date'):
                enriched_text += f"Date de l'événement: {metadata['date']}.\n"
            if metadata.get('activity'):
                enriched_text += f"Activité concernée: {metadata['activity']}.\n"
            if metadata.get('header_path'):
                header_path = ' > '.join(metadata['header_path'])
                enriched_text += f"Contexte hiérarchique: {header_path}.\n"
            
            enriched_text += "\nCONTENU REGROUPÉ DU DOCUMENT:\n"
            
            # Concaténer tous les contenus du groupe
            for chunk in chunks_group:
                enriched_text += f"\n{chunk['content']}\n"
            
            # S'assurer que toutes les métadonnées sont du bon type
            flat_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, (list, tuple)):
                    # Convertir les listes en chaînes
                    flat_metadata[key] = ' > '.join(str(v) for v in value)
                elif isinstance(value, (int, float)):
                    # Convertir les nombres en chaînes
                    flat_metadata[key] = str(value)
                elif isinstance(value, dict):
                    # Ignorer les dictionnaires
                    continue
                else:
                    flat_metadata[key] = value
            
            flat_metadata['source_file'] = file_path
            flat_metadata['chunk_count'] = str(len(chunks_group))
            
            print(f"  Métadonnées finales (aplaties):")
            for key, value in flat_metadata.items():
                print(f"    {key}: {value} (type: {type(value)})")
            
            doc = Document(
                text=enriched_text,
                metadata=flat_metadata,
                excluded_embed_metadata_keys=['source_file'],
                excluded_llm_metadata_keys=['source_file']
            )
            documents.append(doc)
            
            print(f"✓ Document créé avec métadonnées: {' | '.join(f'{k}={v}' for k,v in flat_metadata.items() if v)}")
            print(f"  Contient {len(chunks_group)} chunks regroupés")
            
        except Exception as e:
            print(f"Erreur lors de la création du document groupé: {str(e)}")
            skipped_chunks += len(chunks_group)
    
    if not documents:
        print(f"Warning: Aucun document valide trouvé dans {file_path}")
    else:
        print(f"✓ {len(documents)} documents créés, {skipped_chunks} chunks ignorés")
    
    return documents

def print_index_summary(documents: List[Document]) -> None:
    """Affiche un résumé des documents indexés, organisé par sprint."""
    print("\n=== RÉSUMÉ DE L'INDEXATION ===")
    
    # Organiser les documents par sprint
    sprint_docs = defaultdict(list)
    no_sprint_docs = []
    
    for doc in documents:
        if 'sprint' in doc.metadata and doc.metadata['sprint']:
            sprint_docs[str(doc.metadata['sprint'])].append(doc)  # Convertir en string
        else:
            no_sprint_docs.append(doc)
    
    # Afficher les statistiques par sprint
    print("\nDOCUMENTS PAR SPRINT:")
    def sprint_sort_key(x):
        try:
            return int(x)
        except ValueError:
            return float('inf')
            
    for sprint in sorted(sprint_docs.keys(), key=sprint_sort_key):
        docs = sprint_docs[sprint]
        print(f"\nSprint {sprint}:")
        print(f"  Documents: {len(docs)}")
        
        # Dates
        dates = sorted([d.metadata.get('date') for d in docs if 'date' in d.metadata and d.metadata['date']])
        if dates:
            print(f"  Période: du {dates[0]} au {dates[-1]}")
        
        # Activités
        activities = set(d.metadata.get('activity') for d in docs if 'activity' in d.metadata and d.metadata['activity'])
        if activities:
            print("  Activités:")
            for activity in sorted(activities):
                activity_docs = [d for d in docs if d.metadata.get('activity') == activity]
                print(f"    - {activity} ({len(activity_docs)} documents)")
        
        # Total des chunks
        total_chunks = sum(int(d.metadata.get('chunk_count', 1)) for d in docs)
        print(f"  Total chunks: {total_chunks}")
    
    # Documents sans sprint
    if no_sprint_docs:
        print("\nDOCUMENTS SANS SPRINT:")
        print(f"  Nombre total: {len(no_sprint_docs)}")
        
        # Dates
        dates = sorted([d.metadata.get('date') for d in no_sprint_docs if 'date' in d.metadata and d.metadata['date']])
        if dates:
            print(f"  Période: du {dates[0]} au {dates[-1]}")
        
        # Activités
        activities = set(d.metadata.get('activity') for d in no_sprint_docs if 'activity' in d.metadata and d.metadata['activity'])
        if activities:
            print("  Activités:")
            for activity in sorted(activities):
                activity_docs = [d for d in no_sprint_docs if d.metadata.get('activity') == activity]
                print(f"    - {activity} ({len(activity_docs)} documents)")
        
        # Total des chunks
        total_chunks = sum(int(d.metadata.get('chunk_count', 1)) for d in no_sprint_docs)
        print(f"  Total chunks: {total_chunks}")
    
    # Statistiques globales
    print("\nSTATISTIQUES GLOBALES:")
    print(f"  Nombre total de documents: {len(documents)}")
    print(f"  Nombre de sprints: {len(sprint_docs)}")
    total_chunks = sum(int(d.metadata.get('chunk_count', 1)) for d in documents)
    print(f"  Nombre total de chunks: {total_chunks}")

def load_or_create_index(chunks_dir: str, llm: Any, embed_model: Any, config: dict, force_rebuild: bool = False) -> VectorStoreIndex:
    """
    Charge ou crée un index à partir des chunks FrontMatter.
    """
    persist_dir = Path(chunks_dir).parent / "chroma_db"
    persist_dir.mkdir(exist_ok=True)
    print(f"Chemin de persistance : {persist_dir}")
    
    # Récupérer la liste des métadonnées requises depuis la config
    required_metadata = config.get('indexing', {}).get('required_metadata', ['sprint'])
    print(f"\nMétadonnées requises pour l'indexation : {required_metadata}")
    
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
        documents = load_frontmatter_chunks(file_path, required_metadata)
        all_documents.extend(documents)
    
    # Afficher le résumé de l'indexation
    print_index_summary(all_documents)
    
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