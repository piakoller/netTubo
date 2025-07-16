import streamlit as st
import pandas as pd
import logging
from datetime import datetime
import json
import re

# Import your own modules
from utils import (
    get_patient_ids_for_selection,
    get_available_llm_models_for_patient,
    get_variants_for_patient_and_model,
    extract_recommendation_from_entry,
    save_comparative_evaluation,
    check_if_evaluated,
    PATIENT_DATA
)
from data_loader import load_patient_data

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

st.set_page_config(page_title="NET LLM Comparative Evaluation", page_icon="🧬", layout="wide")
st.title("NET Tumorboard Recommendation - Comparative Expert Evaluation")
st.markdown("---")

if 'df_patients' not in st.session_state:
    st.session_state.df_patients = load_patient_data(PATIENT_DATA)

df_patients = st.session_state.df_patients

if df_patients is not None:
    df_patients = df_patients.dropna(axis=1, how='all')
    df_patients = df_patients.loc[:, ~df_patients.columns.str.contains('^Unnamed')]

if df_patients is None:
    st.error("Patientendaten konnten nicht geladen werden. Bitte prüfe die Datenquelle.")
    st.stop()

def get_patient_summary_text(df_patients, patient_id):
    """Generate a summary text for the patient from the Excel data"""
    if df_patients is None:
        return "Keine Patientendaten verfügbar."
    
    patient_row = df_patients[df_patients["ID"].astype(str) == str(patient_id)]
    if patient_row.empty:
        return "Keine Patientendaten gefunden."
    
    patient_data = patient_row.iloc[0]
    summary_parts = []
    
    # Add key patient information
    for col, val in patient_data.items():
        if pd.notna(val) and str(val).strip():
            summary_parts.append(f"**{col}:** {val}")
    
    return "\n".join(summary_parts)

def render_evaluation_widgets(variant_key: str, patient_id: str, storage_dict: dict, form_key: str):
    """Render evaluation widgets for a specific variant"""
    st.markdown(f"**Evaluation for {variant_key}:**")
    
    # Quality ratings
    col1, col2, col3 = st.columns(3)
    
    with col1:
        clinical_accuracy = st.selectbox(
            "Clinical Accuracy:",
            options=[1, 2, 3, 4, 5],
            key=f"clinical_accuracy_{variant_key}_{patient_id}_{form_key}",
            help="1=Poor, 5=Excellent"
        )
        storage_dict[variant_key]["clinical_accuracy"] = clinical_accuracy
    
    with col2:
        guideline_adherence = st.selectbox(
            "Guideline Adherence:",
            options=[1, 2, 3, 4, 5],
            key=f"guideline_adherence_{variant_key}_{patient_id}_{form_key}",
            help="1=Poor, 5=Excellent"
        )
        storage_dict[variant_key]["guideline_adherence"] = guideline_adherence
    
    with col3:
        overall_quality = st.selectbox(
            "Overall Quality:",
            options=[1, 2, 3, 4, 5],
            key=f"overall_quality_{variant_key}_{patient_id}_{form_key}",
            help="1=Poor, 5=Excellent"
        )
        storage_dict[variant_key]["overall_quality"] = overall_quality
    
    # Comments
    comments = st.text_area(
        "Comments:",
        height=100,
        key=f"comments_{variant_key}_{patient_id}_{form_key}"
    )
    storage_dict[variant_key]["comments"] = comments

def extract_llm_input_sections(llm_input_text: str) -> dict:
    """Extract sections from LLM input text"""
    sections = {
        "system_instruction": "",
        "context_info": "",
        "patient_information": "",
        "attached_documents": ""
    }
    
    try:
        if llm_input_text:
            llm_input_data = json.loads(llm_input_text)
            if isinstance(llm_input_data, dict):
                sections["system_instruction"] = llm_input_data.get("system_instruction", "")
                sections["context_info"] = llm_input_data.get("context_info", "")
                sections["patient_information"] = llm_input_data.get("patient_information", "")
                sections["attached_documents"] = llm_input_data.get("attached_documents", "")
    except:
        sections["system_instruction"] = llm_input_text
    
    return sections

def extract_section(text: str, tag: str) -> str:
    """Extract content between XML-like tags or get remaining content for 'begründung'"""
    if tag in ["beurteilung", "therapieempfehlung"]:
        # Normale Extraktion für Beurteilung und Therapieempfehlung
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        try:
            start = text.index(start_tag) + len(start_tag)
            end = text.index(end_tag)
            return text[start:end].strip()
        except ValueError:
            return ""
    elif tag == "begründung":
        # Für Begründung: Nimm alles nach </therapieempfehlung>
        try:
            split_pos = text.index("</therapieempfehlung>") + len("</therapieempfehlung>")
            return text[split_pos:].strip()
        except ValueError:
            return ""
    return ""

def display_variant_content(variant_data: dict, variant_key: str, patient_id: str):
    """Display content for a variant"""
    entry = variant_data['entry']
    filename = variant_data['filename']
    
    st.caption(f"Source: `{filename}`")
    
    # Extract recommendation components
    final_rec, think_block, raw_output, llm_input = extract_recommendation_from_entry(entry)
    
    # LLM Input
    if llm_input and llm_input.strip() != "{}":
        with st.expander("LLM Input", expanded=False):
            if isinstance(llm_input, str):
                try:
                    llm_input = json.loads(llm_input)
                except json.JSONDecodeError:
                    st.error("Fehler beim Parsen von llm_input als JSON.")
                    llm_input = {}
            
            prompt_text = llm_input.get("prompt_text")
            if prompt_text:
                st.markdown("**Prompt Text:**")
                st.code(f"{prompt_text}")

            patient_info = llm_input.get("patient_information")
            if patient_info:
                st.markdown("**Patient Information:**")
                st.markdown(f"```\n{patient_info}\n```")

            attachments = llm_input.get("attached_documents") or llm_input.get("attachments_used")
            if attachments:
                st.markdown("**Attached Documents:**")
                st.markdown(f"```\n{attachments}\n```")
    else:
        st.info("No LLM Input available.")

    # Raw Output sections
    if raw_output:
        # Extract and display each section
        beurteilung = extract_section(raw_output, "beurteilung")
        therapieempfehlung = extract_section(raw_output, "therapieempfehlung")
        begruendung = extract_section(raw_output, "begründung")

        # Display sections in expandable containers
        with st.expander("Beurteilung", expanded=True):
            st.markdown(beurteilung)
        
        with st.expander("Therapieempfehlung", expanded=True):
            st.markdown(therapieempfehlung)
        
        with st.expander("Begründung", expanded=True):
            st.markdown(begruendung)
    else:
        st.info("No raw output sections available.")
    
    # Final recommendation display (if different from raw output)
    if final_rec and final_rec != raw_output:
        with st.expander("Final Processed Recommendation", expanded=False):
            st.text_area(
                f"Final Recommendation - {variant_key}",
                final_rec,
                height=300,
                disabled=True,
                key=f"display_{variant_key}_{patient_id}_{hash(filename)}"
            )
    
    return final_rec

def get_available_prompt_versions(variants: dict) -> list:
    """Extract available prompt versions from variants"""
    prompt_versions = set()
    
    for variant_key, variant_data in variants.items():
        filename = variant_data['filename']
        approach = variant_data['approach']
        
        if approach == 'SinglePrompt':
            # Extract version from SinglePrompt filenames like "singleprompt_..._prompt_v3_1-1.json"
            match = re.search(r'prompt_(v\d+(?:_[\d-]+)?)\.json', filename)
            if match:
                prompt_versions.add(match.group(1))
    
    # Sort versions naturally (v1, v2, v3_1-1, v3_1-2, etc.)
    def version_sort_key(version):
        # Remove 'v' prefix and split by underscores and hyphens
        parts = version[1:].replace('_', '.').replace('-', '.').split('.')
        return [int(x) if x.isdigit() else x for x in parts]
    
    return sorted(list(prompt_versions), key=version_sort_key)

def filter_variants_by_prompt_versions(variants: dict, selected_versions: list) -> dict:
    """Filter variants based on selected prompt versions"""
    if not selected_versions:
        return variants
    
    filtered_variants = {}
    
    for variant_key, variant_data in variants.items():
        filename = variant_data['filename']
        approach = variant_data['approach']
        include_variant = False
        
        if approach == 'SinglePrompt':
            # Check SinglePrompt filenames
            match = re.search(r'prompt_(v\d+(?:_[\d-]+)?)\.json', filename)
            if match and match.group(1) in selected_versions:
                include_variant = True
        
        if include_variant:
            filtered_variants[variant_key] = variant_data
    
    return filtered_variants

# Initialize session state
if 'expert_name' not in st.session_state:
    st.session_state.expert_name = ""
if 'selected_patient_id' not in st.session_state:
    st.session_state.selected_patient_id = None
if 'selected_prompt_versions' not in st.session_state:
    st.session_state.selected_prompt_versions = []

# --- SIDEBAR ---
st.sidebar.title("Navigation & Settings")
st.session_state.expert_name = st.sidebar.text_input("Your Name/Identifier:", value=st.session_state.expert_name)

patient_id_options = get_patient_ids_for_selection()
if not patient_id_options:
    st.sidebar.error("No patient cases loaded from JSON files.")
    st.stop()

# Initialize selected_patient_id if it's None or not in options
if st.session_state.selected_patient_id is None or st.session_state.selected_patient_id not in patient_id_options:
    st.session_state.selected_patient_id = patient_id_options[0]

selected_patient_id = st.sidebar.selectbox(
    "1. Select Patient Case:", 
    options=patient_id_options,
    index=patient_id_options.index(st.session_state.selected_patient_id),
    key="patient_id_selector"
)
st.session_state.selected_patient_id = selected_patient_id

# Get available LLM models for selected patient
llm_model_options = []
if selected_patient_id:
    llm_model_options = get_available_llm_models_for_patient(selected_patient_id)

if not llm_model_options:
    st.sidebar.error("No LLM models found for this patient.")
    st.stop()

selected_llm_model = st.sidebar.selectbox("2. Select LLM Model:", options=llm_model_options)

# Get variants to determine available prompt versions
if selected_patient_id and selected_llm_model:
    variants = get_variants_for_patient_and_model(selected_patient_id, selected_llm_model)
    available_prompt_versions = get_available_prompt_versions(variants)
    
    if available_prompt_versions:
        # Add prompt version selection
        st.sidebar.markdown("---")
        st.sidebar.markdown("**3. Select Prompt Versions to Compare:**")
        
        # Initialize selected versions if empty or contains unavailable versions
        if not st.session_state.selected_prompt_versions or not all(v in available_prompt_versions for v in st.session_state.selected_prompt_versions):
            st.session_state.selected_prompt_versions = available_prompt_versions.copy()
        
        selected_prompt_versions = st.sidebar.multiselect(
            "Choose which prompt versions to display:",
            options=available_prompt_versions,
            default=st.session_state.selected_prompt_versions,
            key="prompt_version_selector"
        )
        
        # Update session state
        st.session_state.selected_prompt_versions = selected_prompt_versions
        
        # Validate that at least one version is selected
        if not selected_prompt_versions:
            st.sidebar.error("Please select at least one prompt version to display.")
            st.stop()
    else:
        selected_prompt_versions = []
        st.sidebar.warning("No prompt versions found for this patient-LLM combination.")
else:
    selected_prompt_versions = []

# # Check if already evaluated
# if st.session_state.expert_name and selected_patient_id and selected_llm_model:
#     already_evaluated = check_if_evaluated(selected_patient_id, selected_llm_model, st.session_state.expert_name)
#     if already_evaluated:
#         st.sidebar.warning("⚠️ You have already evaluated this patient-LLM combination.")
#     else:
#         st.sidebar.success("✅ Ready for evaluation!")

# --- MAIN CONTENT ---
if selected_patient_id and selected_llm_model and selected_prompt_versions:
    
    # Display patient information
    st.subheader("Patient Information")
    patient_summary = get_patient_summary_text(df_patients, selected_patient_id)
    st.markdown(patient_summary)
    
    st.markdown("---")
    
    # Get variants for this patient and LLM model
    all_variants = get_variants_for_patient_and_model(selected_patient_id, selected_llm_model)
    
    if not all_variants:
        st.error(f"No variants found for Patient {selected_patient_id} and LLM {selected_llm_model}")
        st.stop()
    
    # Filter variants by selected prompt versions
    variants = filter_variants_by_prompt_versions(all_variants, selected_prompt_versions)
    
    if not variants:
        st.warning(f"No variants found for the selected prompt versions: {', '.join(selected_prompt_versions)}")
        st.stop()
    
    form_eval_data_storage = {}
    
    # --- THE SINGLE FORM STARTS HERE ---
    form_key = f"eval_form_comparative_{selected_patient_id}_{selected_llm_model.replace(' ', '_')}_{st.session_state.expert_name.replace(' ', '_')}"
    
    with st.form(key=form_key):
        st.subheader(f"LLM: `{selected_llm_model}` - Recommendation Variants & Your Evaluation")
        
        # Show which prompt versions are currently displayed
        if len(selected_prompt_versions) > 1:
            st.info(f"📋 Displaying prompt versions: **{', '.join(selected_prompt_versions)}**")
        else:
            st.info(f"📋 Displaying prompt version: **{selected_prompt_versions[0]}**")

        # Filter to only show SinglePrompt variants
        sp_variants = {k: v for k, v in variants.items() if v['approach'] == 'SinglePrompt'}
        
        # --- Single-Prompt Section ---
        if sp_variants:
            st.markdown("#### Prompt Variants")
            sp_cols = st.columns(len(sp_variants))
            
            for idx, (variant_key, variant_data) in enumerate(sp_variants.items()):
                with sp_cols[idx]:
                    match = re.search(r'prompt_(v\d+(?:_[\d-]+)?)\.json', variant_data['filename'])
                    prompt_version = f"Prompt {match.group(1)}" if match else "Unknown Prompt"
                    st.markdown(f"**{prompt_version}**")
                   
                    final_rec = display_variant_content(variant_data, variant_key, selected_patient_id)
                    
                    if final_rec and final_rec.strip():
                        form_eval_data_storage[variant_key] = {}
                        render_evaluation_widgets(variant_key, selected_patient_id, form_eval_data_storage, form_key)
                    else:
                        st.info("No recommendation available for evaluation.")
        else:
            st.warning("No Single-Prompt variants found for the selected prompt versions.")
        
        st.markdown(f"\n**💬 Overall Comments for {selected_llm_model} on Patient {selected_patient_id}** (Versions: {', '.join(selected_prompt_versions)})")
        overall_comments_for_llm_patient_input = st.text_area(
            "General comments:",
            height=100, 
            key=f"gen_comments_llm_pat_{selected_patient_id}_{selected_llm_model}",
            help=f"Comments about {selected_llm_model}'s performance across prompt versions: {', '.join(selected_prompt_versions)}"
        )

        # Submit button for the entire form
        versions_text = " & ".join(selected_prompt_versions)
        submitted_form = st.form_submit_button(f"Submit Evaluation: {selected_llm_model} ({versions_text}) - Patient {selected_patient_id}")

        if submitted_form:
            # Check if expert name is provided
            if not st.session_state.expert_name:
                st.error("Please enter your Name/Identifier in the sidebar to submit an evaluation.")
            # Check if there was anything to evaluate
            elif not form_eval_data_storage: 
                st.error("No recommendation variants were found or displayed for evaluation. Cannot submit.")
            else:
                evaluation_payload = {
                    "evaluation_timestamp": datetime.now().isoformat(),
                    "prompt_versions_evaluated": selected_prompt_versions,
                    "evaluations_per_variant": form_eval_data_storage, 
                    "overall_comments_for_llm_patient": overall_comments_for_llm_patient_input,
                    "source_files": {k: v['filename'] for k, v in variants.items() if k in form_eval_data_storage}
                }
                success, saved_filename = save_comparative_evaluation(
                    selected_patient_id,
                    selected_llm_model, 
                    evaluation_payload, 
                    st.session_state.expert_name
                )
                if success:
                    st.success(f"Evaluation submitted successfully! Saved as: **{saved_filename}**")
                    st.balloons()
                    st.rerun() 
                else:
                    st.error("Failed to save evaluation. Check console/logs.")

else:
    st.info("Select a Patient ID, LLM Model, and Prompt Versions from the sidebar, and enter your name to begin.")