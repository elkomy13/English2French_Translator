import streamlit as st
import torch
import os
import time
import json

# Try to import transformers with error handling
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import transformers: {e}")
    st.error("Please install transformers with: pip install transformers")
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None

# Page configuration
st.set_page_config(
    page_title="English to French Translator",
    page_icon="🇫🇷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .translation-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .stTextArea > div > div > textarea {
        font-size: 16px;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_tokenizer():
    """Load the model and tokenizer from local files."""
    if not TRANSFORMERS_AVAILABLE:
        return None, None, None, None
        
    try:
        # Define the model path - adjust this to match your actual path structure
        
        # Alternative paths to try
        possible_paths = [
            "C:\\Users\\youss\\OneDrive\\Desktop\\LLM Tasks\\English2French\\model",
        ]
        
        # Find the correct model path
        actual_model_path = None
        for path in possible_paths:
            if os.path.exists(os.path.join(path, "config.json")):
                actual_model_path = path
                break
        
        if actual_model_path is None:
            # If no standard path found, check current directory for model files
            if os.path.exists("config.json"):
                actual_model_path = "."
            else:
                raise FileNotFoundError("Could not find model files. Please ensure the model is saved correctly.")
        
        st.info(f"Loading model from: {actual_model_path}")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(actual_model_path, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(actual_model_path, local_files_only=True)
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        return model, tokenizer, device, actual_model_path
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please ensure your model files are in the correct directory.")
        return None, None, None, None

def check_model_files():
    """Check if all required model files are present."""
    required_files = [
        "config.json",
        "tokenizer_config.json", 
        "vocab.json",
        "source.spm",
        "target.spm"
    ]
    
    # Check if pytorch model or safetensors exists
    model_files = ["pytorch_model.bin", "model.safetensors"]
    
    # Look for files in current directory and common subdirectories
    search_paths = ['C:\\Users\\youss\\OneDrive\\Desktop\\LLM Tasks\\English2French\\model']

    file_status = {}
    found_path = None
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            path_files = os.listdir(search_path)
            missing_files = []
            
            # Check required files
            for file in required_files:
                if file not in path_files:
                    missing_files.append(file)
            
            # Check for model weights
            has_model_weights = any(mf in path_files for mf in model_files)
            if not has_model_weights:
                missing_files.append("model weights (pytorch_model.bin or model.safetensors)")
            
            if not missing_files:
                found_path = search_path
                file_status[search_path] = {"status": "complete", "files": path_files}
                break
            else:
                file_status[search_path] = {"status": "incomplete", "missing": missing_files, "files": path_files}
    
    return file_status, found_path

def translate_text(text, model, tokenizer, device, max_length=128, num_beams=4):
    """Translate text from English to French."""
    try:
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate translation
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2,
                do_sample=False
            )
        
        # Decode the translation
        translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        return translation
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🇬🇧 ➡️ 🇫🇷 Marine English to French Translator</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Check if transformers is available
    if not TRANSFORMERS_AVAILABLE:
        st.markdown('<div class="error-box">', unsafe_allow_html=True)
        st.error("❌ Transformers library not available")
        st.write("**To fix this issue:**")
        st.code("pip uninstall transformers")
        st.code("pip install transformers")
        st.write("Then restart your Streamlit app.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Check model files first
    file_status, model_path = check_model_files()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Model Status")
        
        # Display transformers status
        if TRANSFORMERS_AVAILABLE:
            st.success("✅ Transformers library loaded")
        else:
            st.error("❌ Transformers library missing")
            return
        
        # Display file status
        if model_path:
            st.success(f"✅ Model found at: `{model_path}`")
            st.markdown('<div class="success-box">All required files are present!</div>', unsafe_allow_html=True)
        else:
            st.error("❌ Model files not found or incomplete")
            st.subheader("📁 File Status:")
            for path, status in file_status.items():
                if os.path.exists(path):
                    st.write(f"**{path}**:")
                    if status["status"] == "incomplete":
                        st.write("❌ Missing files:")
                        for missing in status["missing"]:
                            st.write(f"  - {missing}")
                    st.write("📄 Found files:")
                    for file in status["files"][:10]:  # Show first 10 files
                        st.write(f"  - {file}")
            return
        
        # Advanced settings
        st.subheader("🔧 Translation Settings")
        max_length = st.slider("Max Length", 64, 256, 128, help="Maximum length of translation")
        num_beams = st.slider("Beam Search", 1, 8, 4, help="Number of beams for beam search")
        
        # Model info
        st.subheader("📊 Model Information")
        st.info("**Base Model**: Helsinki-NLP/opus-mt-en-fr")
        st.info("**Fine-tuned on**: KDE4 Dataset")
        st.info("**Task**: English → French Translation")
        
        # Device info
        device_info = "🚀 GPU Available" if torch.cuda.is_available() else "💻 CPU Only"
        st.success(f"**Device**: {device_info}")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🇬🇧 English Input")
        english_text = st.text_area(
            "Enter English text to translate:",
            height=200,
            placeholder="Type your English text here...",
            key="english_input"
        )
        

    with col2:
        st.subheader("🇫🇷 French Translation")
        
        if st.button("🔄 Translate", type="primary", use_container_width=True):
            if english_text.strip():
                # Load model and tokenizer
                with st.spinner("Loading model..."):
                    model, tokenizer, device, actual_path = load_model_and_tokenizer()
                
                if model is None:
                    st.error("Failed to load model. Please check the file structure.")
                    return
                
                # Perform translation
                with st.spinner("Translating..."):
                    start_time = time.time()
                    translation = translate_text(english_text, model, tokenizer, device, max_length, num_beams)
                    end_time = time.time()
                
                if translation:
                    # Display translation
                    st.markdown('<div class="translation-box">', unsafe_allow_html=True)
                    st.write("**Translation:**")
                    st.write(f"*{translation}*")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Translation metrics
                    translation_time = end_time - start_time
                    input_length = len(english_text.split())
                    output_length = len(translation.split())
                    
                    metric_cols = st.columns(3)
                    with metric_cols[0]:
                        st.metric("⏱️ Time", f"{translation_time:.2f}s")
                    with metric_cols[1]:
                        st.metric("📊 Input Words", input_length)
                    with metric_cols[2]:
                        st.metric("📊 Output Words", output_length)
                    
                    # Copy button
                    if st.button("📋 Copy Translation", use_container_width=True):
                        st.write("Translation copied to display above!")
            else:
                st.warning("Please enter some English text to translate.")
        
        # Display placeholder when no translation
        if "translation" not in st.session_state:
            st.markdown('<div class="translation-box">', unsafe_allow_html=True)
            st.write("*Translation will appear here...*")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p>🤖 Powered by MarianMT • Fine-tuned on KDE4 Dataset • Built with Streamlit</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()