from typing import Dict, Any, Optional
from dataclasses import dataclass
import re

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
        self.analysis_template = '''
TASK: Analyze the question and extract ONLY EXPLICITLY mentioned information.
STRICT RULES:
1. NEVER invent or deduce information
2. Set null if information is not explicitly mentioned
3. Reply ONLY in JSON, without additional text
4. Do not add examples or comments
5. Do not add additional fields
EXPECTED EXACT RESPONSE FORMAT:
{{
"sprint": "[number if mentioned, otherwise null]",
"date": "[exact date if mentioned, otherwise null]", 
"activity": "[activity if mentioned, otherwise null]",
"context": "[context if mentioned, otherwise null]",
"enriched_query": "[original query enriched with metadata, e.g. rephrasing the quey+each metadata as a comprehensive query]"
}}
Question to proceed: {query}
        '''

    def preprocess_query(self, query: str) -> QueryMetadata:
        # Analyser la requête avec le LLM
        analysis_prompt = self.analysis_template.format(query=query)
        analysis_response = self.llm.complete(analysis_prompt)
        print(f"Réponse de la pré-analyse: {analysis_response.text}")
        
        try:
            # Nettoyer la réponse pour ne garder que le JSON
            json_content = analysis_response.text.strip()
            # Trouver le début du JSON (première accolade)
            start_idx = json_content.find('{')
            if start_idx != -1:
                json_content = json_content[start_idx:]
            
            # Parser la réponse JSON
            import json
            metadata = json.loads(json_content)
            
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