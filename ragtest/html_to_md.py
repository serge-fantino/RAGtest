from bs4 import BeautifulSoup
import re
from datetime import datetime
import os

def convert_html_to_md(html_file_path, output_dir=None):
    """
    Converts a daily notes HTML file to a structured markdown file.
    
    Args:
        html_file_path: Path to the HTML file
        output_dir: Optional output directory for the markdown file
    Returns:
        str: Path to the generated markdown file
    """
    # Read HTML file
    with open(html_file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file.read(), 'html.parser')

    md_content = []
    
    # Extract title
    title = soup.find('span', id='title-text')
    if title:
        md_content.append(f"# {title.text.strip()}\n")

    # Extract main content
    main_content = soup.find('div', id='main-content')
    if not main_content:
        print(f"Warning: No main content found in {html_file_path}")
        return None

    # Process headers and content
    for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'li']):
        if element.name.startswith('h'):
            level = int(element.name[1])
            header_text = element.text.strip()
            
            # Convert date formats in headers
            date_match = re.search(r'(\d{1,2}\s+[A-Za-z]+\s+\d{4})', header_text)
            if date_match:
                try:
                    date_str = date_match.group(1)
                    date_obj = datetime.strptime(date_str, '%d %b %Y')
                    header_text = header_text.replace(date_str, date_obj.strftime('%Y-%m-%d'))
                except ValueError:
                    pass
            
            md_content.append(f"{'#' * level} {header_text}\n")
        
        elif element.name == 'p':
            text = element.get_text(strip=True)
            if text:
                md_content.append(f"{text}\n\n")
        
        elif element.name == 'ul':
            for li in element.find_all('li', recursive=False):
                is_task = 'inline-task-list' in element.get('class', [])
                task_status = '- [x]' if 'checked' in li.get('class', []) else '- [ ]'
                prefix = task_status if is_task else '-'
                
                text = li.get_text(strip=True)
                if text:
                    md_content.append(f"{prefix} {text}\n")
            md_content.append("\n")

    # Create output file path
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(html_file_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.md")
    else:
        output_path = os.path.splitext(html_file_path)[0] + '.md'

    # Write markdown content
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(''.join(md_content))
    
    print(f"Converted {html_file_path} to {output_path}")
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert HTML daily notes to Markdown')
    parser.add_argument('input_path', help='Path to HTML file or directory')
    parser.add_argument('--output-dir', help='Output directory for markdown files')
    args = parser.parse_args()

    if os.path.isfile(args.input_path):
        convert_html_to_md(args.input_path, args.output_dir)
    elif os.path.isdir(args.input_path):
        for root, _, files in os.walk(args.input_path):
            for file in files:
                if file.endswith('.html'):
                    html_path = os.path.join(root, file)
                    convert_html_to_md(html_path, args.output_dir) 