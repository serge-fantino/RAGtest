import re
from typing import Any
from llama_index.core import VectorStoreIndex
from query_preprocessor import QueryPreprocessor

class HybridPostprocessor:
    """Combine recherche sémantique et par mots-clés."""
    def postprocess_nodes(self, nodes, query_bundle):
        query_str = query_bundle.query_str.lower()
        reranked_nodes = []
        
        print("\n=== Détails du filtrage des nœuds ===")
        
        # Définition des métadonnées importantes à vérifier
        important_metadata = ['sprint', 'date', 'activity', 'context', 'header_path']
        
        for node in nodes:
            score = node.score or 0
            text = node.text.lower()
            metadata = node.metadata
            
            print(f"\nMétadonnées du document:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
            
            # 1. Score basé sur les métadonnées
            metadata_score = 0
            
            for meta_key in important_metadata:
                if meta_key in metadata and metadata[meta_key]:
                    meta_value = str(metadata[meta_key]).lower()
                    
                    # Correspondance exacte dans la requête
                    if meta_value in query_str:
                        metadata_score += 1.0
                        print(f"  ✓ Correspondance exacte {meta_key}: {meta_value} -> +1.0")
                    
                    # Correspondance partielle des mots
                    query_words = query_str.split()
                    meta_words = meta_value.split()
                    matches = sum(1 for w in query_words if any(w in m for m in meta_words))
                    if matches > 0:
                        partial_score = matches * 0.5
                        metadata_score += partial_score
                        print(f"  ✓ Correspondance partielle {meta_key}: {matches} mots -> +{partial_score}")
            
            # 2. Score basé sur les mots-clés dans le texte
            keywords = query_str.split()
            keyword_matches = sum(1 for word in keywords if word in text)
            keyword_score = keyword_matches * 0.1
            
            # Score final combiné avec priorité aux métadonnées
            final_score = score + (metadata_score * 2) + keyword_score
            
            print(f"Scores pour le document:")
            print(f"  - Score initial: {score:.3f}")
            print(f"  - Score métadonnées: {metadata_score:.3f}")
            print(f"  - Score mots-clés: {keyword_score:.3f}")
            print(f"  - Score final: {final_score:.3f}")
            
            node.score = final_score
            reranked_nodes.append(node)
        
        # Trier par score décroissant
        reranked_nodes.sort(key=lambda x: x.score, reverse=True)
        
        # Filtrage intelligent basé sur les scores
        threshold = max(0.5, max(n.score for n in reranked_nodes) * 0.5)
        filtered_nodes = [n for n in reranked_nodes if n.score > threshold]
        
        print(f"\nSeuil de filtrage: {threshold:.3f}")
        print(f"Nombre de nœuds retenus: {len(filtered_nodes)}")
        
        return filtered_nodes if filtered_nodes else reranked_nodes[:3]

class MetadataAwareQueryEngine:
    """Moteur de requête qui prend en compte les métadonnées."""
    def __init__(self, base_engine, preprocessor):
        self.base_engine = base_engine
        self.preprocessor = preprocessor
    
    def query(self, query_str: str) -> Any:
        # Prétraiter la requête
        query_metadata = self.preprocessor.preprocess_query(query_str)
        print("\nAnalyse de la requête:")
        print(f"- Sprint: {query_metadata.sprint}")
        print(f"- Date: {query_metadata.date}")
        print(f"- Activité: {query_metadata.activity}")
        print(f"- Contexte: {query_metadata.context}")
        print(f"\nRequête enrichie: {query_metadata.enriched_query}")
        
        # Au lieu d'utiliser metadata_filters, nous utilisons déjà HybridPostprocessor
        # qui gère le filtrage des métadonnées
        response = self.base_engine.query(query_metadata.enriched_query)
        return response

def create_query_engine(index: VectorStoreIndex, llm: Any, config: dict) -> MetadataAwareQueryEngine:
    """
    Crée un moteur de requête optimisé pour utiliser les métadonnées.
    """
    query_config = config['query_engine']
    preprocessor = QueryPreprocessor(llm)
    
    # Template pour les réponses
    custom_template = """
    Tu es un assistant précis et factuel. Utilise les informations du CONTEXTE ci-dessous pour répondre à la QUESTION.
    Chaque extrait de contexte contient des métadonnées importantes comme le sprint, la date et l'activité.
    Prends en compte ces métadonnées dans ta réponse et cite-les.
    Si tu ne trouves pas l'information dans le contexte, dis-le clairement.
    Ne fais pas de suppositions et reste objectif.

    CONTEXTE:
    {context_str}

    QUESTION:
    {query_str}

    RÉPONSE FACTUELLE:
    """
    
    # Créer le moteur de base
    base_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=query_config.get('similarity_top_k', 10),
        node_postprocessors=[HybridPostprocessor()],
        response_mode=query_config.get('response_mode', "tree_summarize"),
        text_qa_template=custom_template,
        verbose=True
    )
    
    # Wrapper avec notre moteur personnalisé
    return MetadataAwareQueryEngine(base_engine, preprocessor) 