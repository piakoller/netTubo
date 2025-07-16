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

def find_and_convert_pdfs(input_folders, output_base_dir=None, search_subfolders=True):
    if not input_folders:
        print("Keine Eingabeordner angegeben.")
        return

    print(f"Starte Konvertierung von PDFs in MDs...")
    print(f"Eingabeordner: {input_folders}")
    print(f"Unterordner durchsuchen: {'Ja' if search_subfolders else 'Nein'}")
    print("-" * 30)

    # First pass: Count total number of PDF files across all patient folders
    total_pdfs = 0
    for input_folder in input_folders:
        if not os.path.isdir(input_folder):
            continue
        
        # Look for patient_* folders
        for item in os.listdir(input_folder):
            patient_folder_path = os.path.join(input_folder, item)
            if os.path.isdir(patient_folder_path) and item.startswith('patient_'):
                # Count PDFs in this patient folder
                for file in os.listdir(patient_folder_path):
                    if file.lower().endswith('.pdf'):
                        total_pdfs += 1
    
    print(f"Gefunden: {total_pdfs} PDF-Dateien zum Konvertieren")
    print("-" * 30)
    
    # Initialize progress counter
    converted_count = 0

    for input_folder in input_folders:
        if not os.path.isdir(input_folder):
            print(f"Warnung: Eingabeordner '{input_folder}' existiert nicht oder ist kein Verzeichnis. Überspringe.")
            continue

        print(f"Verarbeite Ordner: '{input_folder}'")

        # Look for patient_* folders
        for item in os.listdir(input_folder):
            patient_folder_path = os.path.join(input_folder, item)
            if os.path.isdir(patient_folder_path) and item.startswith('patient_'):
                print(f"\nVerarbeite Patient-Ordner: '{item}'")
                
                # Create mds subfolder in the patient folder
                mds_output_folder = os.path.join(patient_folder_path, 'mds')
                os.makedirs(mds_output_folder, exist_ok=True)
                
                # Process all PDF files in the patient folder
                for file in os.listdir(patient_folder_path):
                    if file.lower().endswith('.pdf'):
                        pdf_path = os.path.join(patient_folder_path, file)
                        md_filename = os.path.splitext(file)[0] + '.md'
                        md_path = os.path.join(mds_output_folder, md_filename)

                        # Show overall progress before conversion
                        converted_count += 1
                        percentage = (converted_count / total_pdfs) * 100
                        print(f"\n📊 Datei {converted_count}/{total_pdfs} ({percentage:.1f}%)")
                        
                        convert_pdf_to_md(pdf_path, md_path)


# --- Konfiguration ---

input_folders_to_process = [
    './clinical_trials_matches/publications'
]

output_markdown_base_directory = None  # Will be set dynamically per patient folder

# Sollen auch Unterordner in den input_folders_to_process durchsucht werden?
search_subdirectories = True

# --- Skript ausführen ---
if __name__ == "__main__":
    find_and_convert_pdfs(input_folders_to_process, search_subfolders=search_subdirectories)
    print("-" * 30)
    print("Konvertierung abgeschlossen.")