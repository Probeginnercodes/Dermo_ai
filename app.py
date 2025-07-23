import os
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Backward compatibility and error prevention
try:
    from huggingface_hub.utils import cached_download 
    
except ImportError:
    try:
        from huggingface_hub import hf_hub_download as cached_download
        import warnings
        warnings.warn("Using hf_hub_download as cached_download replacement", RuntimeWarning)
    except ImportError as e:
        raise ImportError(
            "Failed to import from huggingface_hub. "
            "Ensure huggingface-hub==0.16.4 is in requirements.txt"
        ) from e

import streamlit as st
from PIL import Image
import logging
import torch
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add current directory to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from dermatology_processor.py import MultimodalDermatologyProcessor
except ImportError as e:
    
    logger.error(f"Import failed: {str(e)}")
    st.error("""
    ❌ System initialization failed. Possible causes:
    1. Missing required files (multimodal_processor.py)
    2. Dependency conflicts
    3. Corrupted environment
    """)
    raise

# Streamlit configuration
st.set_page_config(
    page_title="DermoAI - Dermatology Triage Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with error-resistant implementation
custom_css = """
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #f0f2f6;
        margin-bottom: 10px;
    }
    .error-message {
        color: #ff4b4b;
        font-weight: bold;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Initialize session state with robust checks
if 'processor' not in st.session_state:
    st.session_state.processor = None
    st.session_state.processor_loaded = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

@st.cache_resource(show_spinner=False)
def load_processor():
    """Safely load models with comprehensive error handling"""
    try:
        logger.info("Initializing CPU-only models")
        with st.spinner("🚀 Loading DermoAI models (this may take a minute)..."):
            processor = MultimodalDermatologyProcessor(device='cpu')
            logger.info("Models loaded successfully")
            return processor
    except Exception as e:
        logger.exception("Model initialization failed")
        st.error(f"""
        ❌ Critical system error: {str(e)}
        Please try refreshing the page or contact support.
        """)
        return None

def reset_analysis():
    """Safely reset all analysis states"""
    try:
        st.session_state.analysis_results = None
        st.rerun()
    except Exception as e:
        logger.error(f"Reset failed: {str(e)}")
        st.error("Failed to reset analysis. Please refresh the page.")

def main():
    """Main application with enhanced error boundaries"""
    try:
        # Header section
        st.markdown("""
        <div class="main-header">
            <h1>🩺 DermoAI: Dermatology Triage Assistant</h1>
            <p>Preliminary skin condition assessment using AI analysis</p>
        </div>
        """, unsafe_allow_html=True)

        # Medical disclaimer
        st.error("""
        ⚠️ **MEDICAL DISCLAIMER**: This tool provides preliminary assessments only and should NOT replace professional medical consultation.
        - Always consult qualified healthcare professionals
        - Serious symptoms require immediate attention
        - AI may not detect all conditions
        """)

        # Initialize processor with recovery mechanism
        if st.session_state.processor is None and not st.session_state.get('processor_loaded', False):
            st.session_state.processor = load_processor()
            st.session_state.processor_loaded = True
            if st.session_state.processor is None:
                return

        # Patient information sidebar with validation
        with st.sidebar:
            st.header("👤 Patient Information")
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            gender = st.selectbox("Gender", ["Not specified", "Male", "Female", "Other"])
            
            st.subheader("📋 Medical History")
            medical_history = st.text_area(
                "Relevant medical history",
                placeholder="Previous skin conditions, chronic diseases, etc.",
                height=80
            )

        # Main input columns
        col1, col2 = st.columns([1, 1])

        # Image upload section with validation
        with col1:
            st.header("📷 Skin Lesion Image")
            uploaded_image = st.file_uploader(
                "Upload skin lesion photo",
                type=['png', 'jpg', 'jpeg'],
                help="Clear, well-lit photo of the area of concern"
            )

            image_data = None
            if uploaded_image:
                try:
                    image_data = Image.open(uploaded_image)
                    st.image(image_data, caption="Uploaded image", use_column_width=True)
                    
                    if st.session_state.processor:
                        quality = st.session_state.processor.image_processor.analyze_image_quality(image_data)
                        st.markdown(f"""
                        <div class="metric-box">
                            <b>Image Quality:</b> {quality.get('quality', 'unknown').title()}<br>
                            <b>Resolution:</b> {image_data.size[0]}x{image_data.size[1]}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if quality.get('issues'):
                            st.warning(f"⚠️ Image issues: {', '.join(quality['issues'])}")
                except Exception as e:
                    logger.error(f"Image processing error: {str(e)}")
                    st.error("Failed to process image. Please try another file.")

        # Audio and text input section
        with col2:
            st.header("🎙️ Symptom Description")
            
            uploaded_audio = None
            audio_data = None
            if st.toggle("Enable audio input"):
                uploaded_audio = st.file_uploader(
                    "Upload audio description",
                    type=['wav', 'mp3'],
                    help="Describe your symptoms verbally"
                )
                if uploaded_audio:
                    st.audio(uploaded_audio)
            
            text_symptoms = st.text_area(
                "Describe your symptoms:",
                placeholder="Appearance, duration, itching, pain, etc.",
                height=150
            )
            
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                duration = st.selectbox(
                    "Duration",
                    ["<1 week", "1-2 weeks", "2-4 weeks", "1-3 months", ">3 months"]
                )
            with col_d2:
                changes = st.selectbox(
                    "Changes",
                    ["No changes", "Worsening", "Improving", "Changing", "New symptoms"]
                )

        # Analysis button with comprehensive validation
        if st.button("🔬 Analyze Condition", type="primary", use_container_width=True):
            if not (uploaded_image or text_symptoms.strip()):
                st.error("Please provide at least an image or text description")
            else:
                with st.spinner("Analyzing your case (this may take a moment)..."):
                    try:
                        results = st.session_state.processor.process_case(
                            image=image_data if uploaded_image else None,
                            audio=uploaded_audio if uploaded_audio else None,
                            text_data=f"{text_symptoms}. Duration: {duration}. Changes: {changes}",
                            patient_info={
                                'age': age,
                                'gender': gender if gender != "Not specified" else None,
                                'medical_history': medical_history if medical_history else None
                            }
                        )
                        st.session_state.analysis_results = results
                        st.success("Analysis completed!")
                    except Exception as e:
                        logger.exception("Analysis failed")
                        st.error(f"""
                        Analysis failed: {str(e)}
                        Please try again with different inputs.
                        """)

        # Results display with error boundaries
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            try:
                st.markdown("## 📊 Analysis Results")

                cols = st.columns(3)
                cols[0].metric("Urgency Level", results.get('urgency', 'unknown').title())
                cols[1].metric("Confidence", f"{results.get('confidence', 0):.0%}")
                cols[2].metric(
                    "Data Sources Used",
                    f"{sum([bool(results.get(k)) for k in ['image_analysis', 'audio_analysis', 'text_analysis']])}/3"
                )

                if results.get('clinical_assessment'):
                    st.subheader("🏥 Clinical Notes")
                    st.info(results['clinical_assessment'].get('clinical_description', 'No description'))

                    if results['clinical_assessment'].get('differential_diagnoses'):
                        st.subheader("🎯 Possible Conditions")
                        for dx in results['clinical_assessment']['differential_diagnoses']:
                            st.write(f"- {dx.get('condition', 'Unknown')} ({dx.get('likelihood', 'unknown')} likelihood)")

                if results.get('recommendations'):
                    st.subheader("💡 Recommendations")
                    for rec in results['recommendations']:
                        st.write(f"- {rec}")

            except Exception as e:
                logger.error(f"Results display failed: {str(e)}")
                st.error("Failed to display results. Please re-run the analysis.")

        if st.button("🔄 Start New Analysis", use_container_width=True):
            reset_analysis()

    except Exception as e:
        logger.critical(f"Application crash: {str(e)}")
        st.error("""
        💥 Critical application error
        Please refresh the page or try again later.
        """)

# Footer with version info
st.markdown("---")
st.caption(f"""
DermoAI v1.0 | Python {sys.version.split()[0]} | Torch {torch.__version__} | Streamlit {st.__version__}
""")

if __name__ == "__main__":
    # Hugging Face Spaces compatibility
    if os.environ.get('IS_HF_SPACE') == '1':
        os.system(f"streamlit run {__file__} --server.port=$PORT --server.address=0.0.0.0")
    else:
        main()