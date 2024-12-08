from bs4 import BeautifulSoup
import openai
from typing import Dict, List, Tuple
import numpy as np

class DocumentPreprocessor:
    def __init__(self, llm_client):
        self.llm = llm_client
        
    def parse_html(self, html_content: str) -> List[Dict]:
        """Parse le HTML en structure hiérarchique."""
        soup = BeautifulSoup(html_content, 'html.parser')
        chunks = []
        
        # Parcours hiérarchique (mois/sprint/semaine/jour)
        for time_block in soup.find_all(['div', 'section'], class_=['month', 'sprint', 'week', 'day']):
            # Pour chaque bloc temporel, extraire les sections
            for section in time_block.find_all(['div', 'section'], class_=['todo', 'meeting', 'backlog']):
                chunks.extend(self._process_section(section, time_block))
                
        return chunks
    
    def _process_section(self, section, time_context) -> List[Dict]:
        """Traite une section et crée des chunks."""
        chunks = []
        
        # Extraction du contexte temporel
        time_info = {
            'period': time_context.get('class', [''])[0],
            'date': time_context.get('data-date', ''),
        }
        
        # Pour chaque item d'action ou note
        for item in section.find_all(['li', 'p']):
            chunk = {
                'content': item.get_text(strip=True),
                'time_context': time_info,
                'section_type': section.get('class', [''])[0],
                'metadata': {}
            }
            
            if len(chunk['content']) > 20:  # Filtre minimal initial
                chunks.append(chunk)
                
        return chunks
    
    def evaluate_chunk(self, chunk: Dict) -> Tuple[float, Dict]:
        """Évalue la qualité d'un chunk via LLM."""
        prompt = f"""Évaluez ce segment d'information et retournez un JSON avec:
        - score_qualite (1-5): évaluation globale
        - autonomie (1-5): compréhensible sans contexte
        - pertinence (1-5): utilité de l'information
        - coherence (1-5): unité thématique
        - metadata: {{
            "themes": [liste des thèmes],
            "projets": [projets mentionnés],
            "type_activite": type principal d'activité,
            "personnes": [personnes mentionnées]
        }}
        
        Segment à évaluer (contexte: {chunk['section_type']}, période: {chunk['time_context']['period']}):
        {chunk['content']}
        """
        
        evaluation = self.llm.complete(prompt)  # Implémentation à adapter selon votre LLM
        eval_data = eval(evaluation)  # Parsing sécurisé à implémenter
        
        score = np.mean([
            eval_data['score_qualite'],
            eval_data['autonomie'],
            eval_data['pertinence'],
            eval_data['coherence']
        ])
        
        return score, eval_data['metadata']
    
    def process_document(self, html_content: str, min_score: float = 3.5) -> List[Dict]:
        """Traitement complet du document."""
        # Extraction initiale
        chunks = self.parse_html(html_content)
        processed_chunks = []
        
        # Évaluation et enrichissement
        for chunk in chunks:
            score, metadata = self.evaluate_chunk(chunk)
            if score >= min_score:
                chunk['quality_score'] = score
                chunk['metadata'] = metadata
                processed_chunks.append(chunk)
        
        # Optimisation finale
        optimized_chunks = self._optimize_chunks(processed_chunks)
        return optimized_chunks
    
    def _optimize_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Optimise les chunks (fusion/séparation selon besoin)."""
        optimized = []
        buffer = []
        
        for chunk in chunks:
            if len(buffer) == 0:
                buffer.append(chunk)
                continue
            
            # Si les chunks sont thématiquement liés et courts
            if (set(buffer[-1]['metadata']['themes']) & set(chunk['metadata']['themes']) and
                len(buffer[-1]['content']) < 100 and len(chunk['content']) < 100):
                # Fusion
                buffer[-1]['content'] += ' ' + chunk['content']
                buffer[-1]['metadata']['themes'] = list(
                    set(buffer[-1]['metadata']['themes'] + chunk['metadata']['themes'])
                )
            else:
                if buffer:
                    optimized.extend(buffer)
                buffer = [chunk]
        
        optimized.extend(buffer)
        return optimized

# Exemple d'utilisation
if __name__ == "__main__":
    preprocessor = DocumentPreprocessor(llm_client=None)  # Initialiser avec votre client LLM
    
    with open('journal.html', 'r') as f:
        html_content = f.read()
    
    processed_data = preprocessor.process_document(html_content)
    
    # Les chunks résultants sont prêts pour l'indexation RAG
    # Chaque chunk contient :
    # - Le contenu original
    # - Le score de qualité
    # - Les métadonnées enrichies
    # - Le contexte temporel
