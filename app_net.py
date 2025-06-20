import streamlit as st
import pandas as pd
import logging
from datetime import datetime
import json

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
            # parse llm_input string to dict if it's a JSON string
            if isinstance(llm_input, str):
                import json
                try:
                    llm_input = json.loads(llm_input)
                except json.JSONDecodeError:
                    st.error("Fehler beim Parsen von llm_input als JSON.")
                    llm_input = {}

            # get inner content from 'llm_input' field
            # llm_input_sections = llm_input.get("prompt_text", {})
            # print(llm_input_sections)
            
            prompt_text = llm_input.get("prompt_text")
            if prompt_text:
                st.markdown("**Prompt Text:**")
                st.code(f"```\n{prompt_text}\n```")

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
    
    # Think Block / Raw Output
    if think_block:
        with st.expander("Raw LLM Output", expanded=False):
            st.markdown(f"```\n{think_block}\n```")
    else:
        st.info("No raw output available.")
    
    # Final recommendation display
    st.text_area(
        f"Final Recommendation - {variant_key}",
        final_rec,
        height=300,
        disabled=True,
        key=f"display_{variant_key}_{patient_id}_{hash(filename)}"
    )
    
    return final_rec

# Initialize session state
if 'expert_name' not in st.session_state:
    st.session_state.expert_name = ""
if 'selected_patient_id' not in st.session_state:
    st.session_state.selected_patient_id = None

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

# Check if already evaluated
if st.session_state.expert_name and selected_patient_id and selected_llm_model:
    already_evaluated = check_if_evaluated(selected_patient_id, selected_llm_model, st.session_state.expert_name)
    if already_evaluated:
        st.sidebar.warning("⚠️ You have already evaluated this patient-LLM combination.")
    else:
        st.sidebar.success("✅ Ready for evaluation!")

# --- MAIN CONTENT ---
if selected_patient_id and selected_llm_model:
    
    # Display patient information
    st.subheader("Patient Information")
    patient_summary = get_patient_summary_text(df_patients, selected_patient_id)
    st.markdown(patient_summary)
    
    st.markdown("---")
    
    # Get variants for this patient and LLM model
    variants = get_variants_for_patient_and_model(selected_patient_id, selected_llm_model)
    
    if not variants:
        st.error(f"No variants found for Patient {selected_patient_id} and LLM {selected_llm_model}")
        st.stop()
    
    form_eval_data_storage = {}
    
    # --- THE SINGLE FORM STARTS HERE ---
    form_key = f"eval_form_comparative_{selected_patient_id}_{selected_llm_model.replace(' ', '_')}_{st.session_state.expert_name.replace(' ', '_')}"
    
    with st.form(key=form_key):
        st.subheader(f"LLM: `{selected_llm_model}` - Recommendation Variants & Your Evaluation")

        # Group variants by approach
        sp_variants = {k: v for k, v in variants.items() if v['approach'] == 'SinglePrompt'}
        ma_variants = {k: v for k, v in variants.items() if v['approach'] == 'MultiAgent'}
        
        # --- Single-Prompt Section ---
        if sp_variants:
            st.markdown("#### Single-Prompt Approach")
            sp_cols = st.columns(len(sp_variants))
            
            for idx, (variant_key, variant_data) in enumerate(sp_variants.items()):
                with sp_cols[idx]:
                    modifier = "Modified" if variant_data['is_modified'] else "Standard"
                    st.markdown(f"**{modifier} Clinical Info**")
                    
                    final_rec = display_variant_content(variant_data, variant_key, selected_patient_id)
                    
                    if final_rec and final_rec.strip():
                        form_eval_data_storage[variant_key] = {}
                        render_evaluation_widgets(variant_key, selected_patient_id, form_eval_data_storage, form_key)
                    else:
                        st.info("No recommendation available for evaluation.")
        
        # --- Multi-Agent Section ---
        if ma_variants:
            st.markdown("#### Multi-Agent Approach")
            ma_cols = st.columns(len(ma_variants))
            
            for idx, (variant_key, variant_data) in enumerate(ma_variants.items()):
                with ma_cols[idx]:
                    modifier = "Modified" if variant_data['is_modified'] else "Standard"
                    st.markdown(f"**{modifier} Clinical Info**")
                    
                    final_rec = display_variant_content(variant_data, variant_key, selected_patient_id)
                    
                    if final_rec and final_rec.strip():
                        form_eval_data_storage[variant_key] = {}
                        render_evaluation_widgets(variant_key, selected_patient_id, form_eval_data_storage, form_key)
                    else:
                        st.info("No recommendation available for evaluation.")
        
        st.markdown(f"\n**💬 Overall Comments for this LLM's ({selected_llm_model}) Performance on Patient {selected_patient_id}**")
        overall_comments_for_llm_patient_input = st.text_area(
            "General comments:",
            height=100, 
            key=f"gen_comments_llm_pat_{selected_patient_id}_{selected_llm_model}"
        )

        # Submit button for the entire form
        submitted_form = st.form_submit_button(f"Submit All Evaluations for LLM: {selected_llm_model} on Patient: {selected_patient_id}")

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
    st.info("Select a Patient ID and LLM Model from the sidebar, and enter your name to begin.")