import os
from docling.document_converter import DocumentConverter

def convert_pdf_to_md(pdf_path, md_path):
    try:
        print(f"Starte Konvertierung: '{os.path.basename(pdf_path)}'...")
        
        # Initialize docling converter
        converter = DocumentConverter()
        
        print(f"  → PDF wird gelesen und analysiert...")
        # Convert PDF to document
        result = converter.convert(pdf_path)
        
        print(f"  → Text wird extrahiert...")
        # Extract markdown text
        markdown_text = result.document.export_to_markdown()

        print(f"  → Markdown-Datei wird gespeichert...")
        # Stelle sicher, dass das Zielverzeichnis existiert
        os.makedirs(os.path.dirname(md_path), exist_ok=True)

        # Speichere den extrahierten Text als MD-Datei
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)

        print(f"✅ Erfolgreich konvertiert: '{os.path.basename(pdf_path)}' → '{os.path.basename(md_path)}'")

    except FileNotFoundError:
        print(f"❌ Fehler: Die Datei '{pdf_path}' wurde nicht gefunden.")
    except Exception as e:
        print(f"❌ Ein unerwarteter Fehler ist aufgetreten bei '{os.path.basename(pdf_path)}': {e}")

def find_and_convert_pdfs(input_folders, output_base_dir, search_subfolders=True):
    if not input_folders:
        print("Keine Eingabeordner angegeben.")
        return

    print(f"Starte Konvertierung von PDFs in MDs...")
    print(f"Eingabeordner: {input_folders}")
    print(f"Zielordner für MDs: {output_base_dir}")
    print(f"Unterordner durchsuchen: {'Ja' if search_subfolders else 'Nein'}")
    print("-" * 30)

    # Stelle sicher, dass der Basis-Zielordner existiert
    os.makedirs(output_base_dir, exist_ok=True)

    # First pass: Count total number of PDF files
    total_pdfs = 0
    for input_folder in input_folders:
        if not os.path.isdir(input_folder):
            continue
        
        if search_subfolders:
            for root, dirs, files in os.walk(input_folder):
                total_pdfs += sum(1 for file in files if file.lower().endswith('.pdf'))
        else:
            try:
                for file in os.listdir(input_folder):
                    pdf_path = os.path.join(input_folder, file)
                    if os.path.isfile(pdf_path) and file.lower().endswith('.pdf'):
                        total_pdfs += 1
            except Exception:
                pass
    
    print(f"Gefunden: {total_pdfs} PDF-Dateien zum Konvertieren")
    print("-" * 30)
    
    # Initialize progress counter
    converted_count = 0

    for input_folder in input_folders:
        if not os.path.isdir(input_folder):
            print(f"Warnung: Eingabeordner '{input_folder}' existiert nicht oder ist kein Verzeichnis. Überspringe.")
            continue

        print(f"Verarbeite Ordner: '{input_folder}'")

        # Bestimme den Namen des Eingabeordners, um ihn im Zielpfad zu verwenden
        input_folder_name = os.path.basename(input_folder)

        if search_subfolders:
            # Durchsuche den Ordnerbaum
            for root, dirs, files in os.walk(input_folder):
                # Bestimme den relativen Pfad vom ursprünglichen Eingabeordner zum aktuellen Unterordner
                relative_path = os.path.relpath(root, input_folder)

                for file in files:
                    if file.lower().endswith('.pdf'):
                        pdf_path = os.path.join(root, file)
                        # Erstelle den entsprechenden Zielpfad im Ausgabeordner
                        # output_base_dir / input_folder_name / relative_path / pdf_filename.md
                        md_filename = os.path.splitext(file)[0] + '.md'
                        output_folder = os.path.join(output_base_dir, input_folder_name, relative_path)
                        md_path = os.path.join(output_folder, md_filename)

                        # Show overall progress before conversion
                        converted_count += 1
                        percentage = (converted_count / total_pdfs) * 100
                        print(f"\n📊 Datei {converted_count}/{total_pdfs} ({percentage:.1f}%)")
                        
                        convert_pdf_to_md(pdf_path, md_path)
        else:
            # Nur Dateien im obersten Level des Eingabeordners durchsuchen
            try:
                for file in os.listdir(input_folder):
                    pdf_path = os.path.join(input_folder, file)
                    if os.path.isfile(pdf_path) and file.lower().endswith('.pdf'):
                        # Erstelle den entsprechenden Zielpfad im Ausgabeordner
                        # output_base_dir / input_folder_name / pdf_filename.md
                        md_filename = os.path.splitext(file)[0] + '.md'
                        output_folder = os.path.join(output_base_dir, input_folder_name)
                        md_path = os.path.join(output_folder, md_filename)

                        # Show overall progress before conversion
                        converted_count += 1
                        percentage = (converted_count / total_pdfs) * 100
                        print(f"\n📊 Datei {converted_count}/{total_pdfs} ({percentage:.1f}%)")
                        
                        convert_pdf_to_md(pdf_path, md_path)
            except Exception as e:
                 print(f"Fehler beim Auflisten von Dateien in '{input_folder}': {e}")


# --- Konfiguration ---

input_folders_to_process = [
    '../knowledge_base'
]

output_markdown_base_directory = '../mds'

# Sollen auch Unterordner in den input_folders_to_process durchsucht werden?
search_subdirectories = True

# --- Skript ausführen ---
if __name__ == "__main__":
    find_and_convert_pdfs(input_folders_to_process, output_markdown_base_directory, search_subdirectories)
    print("-" * 30)
    print("Konvertierung abgeschlossen.")