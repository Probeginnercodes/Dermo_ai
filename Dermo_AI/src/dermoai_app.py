import streamlit as st
from PIL import Image
import logging
import torch
from multimodal_processor import MultimodalDermatologyProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit configuration
st.set_page_config(
    page_title="DermoAI - Dermatology Triage Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

@st.cache_resource
def load_processor():
    """Load models with CPU-only configuration"""
    try:
        logger.info("Initializing CPU-only models")
        st.info("🚀 Loading DermoAI models (CPU optimized version)...")
        processor = MultimodalDermatologyProcessor(device='cpu')
        st.success("Models loaded successfully!")
        return processor
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        logger.exception("Model initialization error")
        return None

def reset_analysis():
    """Clear previous analysis results"""
    st.session_state.analysis_results = None
    st.rerun()

def main():
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

    # Initialize processor (CPU-only)
    if st.session_state.processor is None:
        st.session_state.processor = load_processor()
        if st.session_state.processor is None:
            return  # Stop execution if models failed to load

    # Patient information sidebar
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

    # Image upload section
    with col1:
        st.header("📷 Skin Lesion Image")
        uploaded_image = st.file_uploader(
            "Upload skin lesion photo",
            type=['png', 'jpg', 'jpeg'],
            help="Clear, well-lit photo of the area of concern"
        )

        if uploaded_image:
            image_data = Image.open(uploaded_image)
            st.image(image_data, caption="Uploaded image", use_column_width=True)
            
            # Image quality analysis
            quality = st.session_state.processor.image_processor.analyze_image_quality(image_data)
            st.markdown("""
            <div class="metric-box">
                <b>Image Quality:</b> {quality}<br>
                <b>Resolution:</b> {width}x{height}
            </div>
            """.format(
                quality=quality.get('quality', 'unknown').title(),
                width=image_data.size[0],
                height=image_data.size[1]
            ), unsafe_allow_html=True)
            
            if quality.get('issues'):
                st.warning(f"⚠️ Image issues: {', '.join(quality['issues'])}")

    # Audio and text input section
    with col2:
        st.header("🎙️ Symptom Description")
        
        # Audio upload
        uploaded_audio = st.file_uploader(
            "Upload audio description (optional)",
            type=['wav', 'mp3'],
            help="Describe your symptoms verbally"
        )
        if uploaded_audio:
            st.audio(uploaded_audio)
        
        # Text input
        text_symptoms = st.text_area(
            "Describe your symptoms:",
            placeholder="Appearance, duration, itching, pain, etc.",
            height=150
        )
        
        # Duration and changes
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

    # Analysis button
    if st.button("🔬 Analyze Condition", type="primary"):
        if not (uploaded_image or text_symptoms.strip()):
            st.error("Please provide at least an image or text description")
        else:
            with st.spinner("Analyzing your case..."):
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
                    st.error(f"Analysis failed: {str(e)}")
                    logger.error(f"Analysis error: {str(e)}")

    # Display results
    if st.session_state.analysis_results:
     results = st.session_state.analysis_results

    st.markdown("## 📊 Analysis Results")

    # Metrics row
    cols = st.columns(3)
    cols[0].metric("Urgency Level", results.get('urgency', 'unknown').title())
    cols[1].metric("Confidence", f"{results.get('confidence', 0):.0%}")
    cols[2].metric(
        "Data Sources Used",
        f"{sum([bool(results.get(k)) for k in ['image_analysis', 'audio_analysis', 'text_analysis']])}/3"
    )

    # Clinical assessment
    if results.get('clinical_assessment'):
        st.subheader("🏥 Clinical Notes")
        st.info(results['clinical_assessment'].get('clinical_description', 'No description'))

        if results['clinical_assessment'].get('differential_diagnoses'):
            st.subheader("🎯 Possible Conditions")
            for dx in results['clinical_assessment']['differential_diagnoses']:
                st.write(f"- {dx.get('condition', 'Unknown')} ({dx.get('likelihood', 'unknown')} likelihood)")

    # Recommendations
    if results.get('recommendations'):
        st.subheader("💡 Recommendations")
        for rec in results['recommendations']:
            st.write(f"- {rec}")

    # Reset button
    if st.button("🔄 Start New Analysis"):
        reset_analysis()

# Footer
st.markdown("---")
st.caption("""
DermoAI - For educational purposes only | Not a substitute for professional medical advice
""")

if __name__ == "__main__":
    main()