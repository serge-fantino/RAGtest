from typing import Dict, Any, Optional
from dataclasses import dataclass

import yaml

@dataclass
class QueryMetadata:
    sprint: Optional[str] = None
    date: Optional[str] = None
    activity: Optional[str] = None
    context: Optional[str] = None
    original_query: str = ""
    enriched_query: str = ""

class QueryPreprocessor:
    def __init__(self, llm):
        self.llm = llm
        self.analysis_template = """
        Analyse la question suivante et extrait UNIQUEMENT les informations EXPLICITEMENT mentionnées.
        
        RÈGLES STRICTES:
        1. Ne JAMAIS inventer ou déduire d'informations
        2. Si une information n'est pas explicitement mentionnée, mettre null
        3. Ne pas essayer de deviner des dates, activités ou contextes
        4. Ne répondre que si l'information est littéralement présente dans la question
        
        Question: {query}
        
        Format YAML attendu:
        sprint: [uniquement le numéro si explicitement mentionné, sinon null]
        date: [date exacte si mentionnée, sinon null]
        activity: [activité si explicitement mentionnée, sinon null]
        context: [contexte si explicitement mentionné, sinon null]
        query_focus: [sujet principal de la question, en quelques mots]
        """

    def preprocess_query(self, query: str) -> QueryMetadata:
        # Analyser la requête avec le LLM
        analysis_prompt = self.analysis_template.format(query=query)
        analysis_response = self.llm.complete(analysis_prompt)
        
        try:
            # Parser la réponse YAML
            metadata = yaml.safe_load(analysis_response.text)
            
            # Construire une requête enrichie
            enriched_parts = []
            if metadata.get('sprint'):
                enriched_parts.append(f"Dans le sprint {metadata['sprint']}")
            if metadata.get('date'):
                enriched_parts.append(f"à la date {metadata['date']}")
            if metadata.get('activity'):
                enriched_parts.append(f"concernant l'activité {metadata['activity']}")
            if metadata.get('context'):
                enriched_parts.append(f"dans le contexte {metadata['context']}")
            if metadata.get('query_focus'):
                enriched_parts.append(f"spécifiquement sur {metadata['query_focus']}")
            
            enriched_query = f"{query} ({' '.join(enriched_parts)})"
            
            return QueryMetadata(
                sprint=metadata.get('sprint'),
                date=metadata.get('date'),
                activity=metadata.get('activity'),
                context=metadata.get('context'),
                original_query=query,
                enriched_query=enriched_query
            )
            
        except Exception as e:
            print(f"Erreur lors du preprocessing de la requête: {e}")
            return QueryMetadata(original_query=query, enriched_query=query) 