from datetime import datetime
import re
from typing import List, Dict
from dataclasses import dataclass
import os

@dataclass
class Metadata:
    date: datetime = None
    sprint: int = None
    week: str = None
    activity: str = None
    parent_headers: List[str] = None

@dataclass
class DocumentChunk:
    content: str
    metadata: Metadata
    source_file: str
    header_path: List[str]  # Chemin complet des titres jusqu'au contenu

class DailyNotesIndex:
    def __init__(self):
        self.chunks: List[DocumentChunk] = []
    
    def process_markdown_file(self, md_file_path: str) -> List[DocumentChunk]:
        """
        Process a markdown file into contextual chunks with rich metadata.
        """
        with open(md_file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Structure pour suivre le contexte courant
        current_context = {
            'headers': [],
            'sprint': None,
            'date': None,
            'week': None
        }
        
        chunks = []
        current_chunk = []
        
        for line in content.split('\n'):
            # Détection des headers et mise à jour du contexte
            if line.startswith('#'):
                if current_chunk:  # Sauvegarder le chunk précédent
                    chunks.append(self._create_chunk(
                        '\n'.join(current_chunk),
                        current_context.copy(),
                        md_file_path
                    ))
                    current_chunk = []
                
                level = len(re.match(r'^#+', line).group())
                text = line.lstrip('#').strip()
                
                # Mise à jour du contexte
                current_context['headers'] = current_context['headers'][:level-1] + [text]
                
                # Extraction des métadonnées du header
                self._update_context_from_header(text, current_context)
                
            else:
                # Accumulation du contenu avec un minimum de lignes
                if line.strip():
                    current_chunk.append(line)
                    if len(current_chunk) >= 3:  # Taille minimum pour un chunk
                        chunks.append(self._create_chunk(
                            '\n'.join(current_chunk),
                            current_context.copy(),
                            md_file_path
                        ))
                        current_chunk = []
        
        # Dernier chunk
        if current_chunk:
            chunks.append(self._create_chunk(
                '\n'.join(current_chunk),
                current_context.copy(),
                md_file_path
            ))
        
        self.chunks.extend(chunks)
        return chunks
    
    def _update_context_from_header(self, header: str, context: Dict):
        """Extrait les métadonnées d'un header."""
        # Sprint detection
        sprint_match = re.search(r'Sprint\s+(\d+)', header)
        if sprint_match:
            context['sprint'] = int(sprint_match.group(1))
        
        # Date detection
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', header)
        if date_match:
            context['date'] = datetime.strptime(date_match.group(1), '%Y-%m-%d')
        
        # Week detection
        week_match = re.search(r'Week of (.*?)(?=\s*/|$)', header)
        if week_match:
            context['week'] = week_match.group(1).strip()
    
    def _create_chunk(self, content: str, context: Dict, source_file: str) -> DocumentChunk:
        """Crée un chunk avec ses métadonnées."""
        metadata = Metadata(
            date=context.get('date'),
            sprint=context.get('sprint'),
            week=context.get('week'),
            activity=context['headers'][-1] if context['headers'] else None,
            parent_headers=context['headers'][:-1] if context['headers'] else []
        )
        
        return DocumentChunk(
            content=content,
            metadata=metadata,
            source_file=source_file,
            header_path=context['headers']
        )

    def get_chunks_for_embedding(self) -> List[Dict]:
        """
        Prépare les chunks pour l'embedding avec un texte enrichi.
        """
        embedding_chunks = []
        for chunk in self.chunks:
            # Construction d'un contexte riche pour l'embedding
            context_prefix = []
            if chunk.metadata.sprint:
                context_prefix.append(f"Sprint {chunk.metadata.sprint}")
            if chunk.metadata.date:
                context_prefix.append(f"Date: {chunk.metadata.date.strftime('%Y-%m-%d')}")
            if chunk.metadata.activity:
                context_prefix.append(f"Activity: {chunk.metadata.activity}")
            
            # Combine context and content
            enriched_text = f"{' | '.join(context_prefix)}\n{chunk.content}"
            
            embedding_chunks.append({
                'text': enriched_text,
                'metadata': chunk.metadata.__dict__,
                'source': chunk.source_file,
                'header_path': chunk.header_path
            })
        
        return embedding_chunks

def save_chunks_to_frontmatter(chunks: List[Dict], output_dir: str, source_file: str):
    """
    Sauvegarde les chunks au format FrontMatter (YAML + Markdown).
    Chaque chunk commence par ses métadonnées en YAML entre '---',
    suivi du contenu en markdown.
    """
    from pathlib import Path
    import yaml
    
    # Créer le répertoire de sortie si nécessaire
    os.makedirs(output_dir, exist_ok=True)
    
    # Créer un nom de fichier basé sur le fichier source
    source_name = Path(source_file).stem
    output_file = os.path.join(output_dir, f"{source_name}_chunks.md")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks, 1):
            # Préparer les métadonnées
            metadata = chunk['metadata'].copy()
            if metadata['date']:
                metadata['date'] = metadata['date'].strftime('%Y-%m-%d')
            
            # Ajouter des métadonnées supplémentaires utiles
            metadata.update({
                'chunk_id': i,
                'source_file': chunk['source'],
                'header_path': chunk['header_path']
            })
            
            # Écrire le bloc FrontMatter
            f.write('---\n')
            yaml.dump(metadata, f, allow_unicode=True, sort_keys=False)
            f.write('---\n\n')
            
            # Écrire le contenu
            f.write(chunk['text'])
            f.write('\n\n')
            
            # Séparateur entre les chunks
            f.write('---\n\n')
    
    return output_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Index markdown files and generate chunks')
    parser.add_argument('input_path', help='Path to markdown file or directory')
    parser.add_argument('--output-dir', help='Output directory for chunk files', required=True)
    args = parser.parse_args()
    
    indexer = DailyNotesIndex()
    processed_files = []
    
    # Traitement des fichiers
    if os.path.isfile(args.input_path):
        if args.input_path.endswith('.md'):
            indexer.process_markdown_file(args.input_path)
            processed_files.append(args.input_path)
    elif os.path.isdir(args.input_path):
        for root, _, files in os.walk(args.input_path):
            for file in files:
                if file.endswith('.md'):
                    md_path = os.path.join(root, file)
                    indexer.process_markdown_file(md_path)
                    processed_files.append(md_path)
    
    # Génération et sauvegarde des chunks
    if indexer.chunks:
        print(f"\nProcessing {len(processed_files)} files...")
        
        # Grouper les chunks par fichier source
        chunks_by_file = {}
        for chunk in indexer.get_chunks_for_embedding():
            source = chunk['source']
            if source not in chunks_by_file:
                chunks_by_file[source] = []
            chunks_by_file[source].append(chunk)
        
        # Sauvegarder chaque groupe de chunks
        for source_file, chunks in chunks_by_file.items():
            output_file = save_chunks_to_frontmatter(chunks, args.output_dir, source_file)
            print(f"Generated chunks for {source_file} -> {output_file}")
            print(f"  Number of chunks: {len(chunks)}")
        
        print(f"\nTotal chunks generated: {len(indexer.chunks)}")
        print(f"Output directory: {os.path.abspath(args.output_dir)}")
    else:
        print("No markdown files processed.")