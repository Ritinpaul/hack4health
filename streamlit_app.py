import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import plotly.graph_objects as go
import time

from src.model import create_model

st.set_page_config(
    page_title="DentalAI - Caries Detection",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #64748b;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background: linear-gradient(120deg, #2563eb, #7c3aed);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        border-radius: 0.5rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .result-box {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 4px solid #2563eb;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fef3c7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
    }
    .success-box {
        background: #d1fae5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #10b981;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🦷 DentalAI - Intelligent Caries Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced AI-powered dental caries segmentation using deep learning</p>', unsafe_allow_html=True)

# Sidebar with better styling
with st.sidebar:
    st.markdown("### ⚙️ Configuration Panel")
    st.markdown("---")
    
    device = st.selectbox(
        "🖥️ Processing Device",
        ['cuda' if torch.cuda.is_available() else 'cpu', 'cpu'],
        help="Select GPU (CUDA) for faster processing if available"
    )
    
    confidence_threshold = st.slider(
        "🎯 Detection Threshold",
        0.0, 1.0, 0.5, 0.05,
        help="Adjust sensitivity: Lower = more detections, Higher = only confident detections"
    )
    
    st.markdown("---")
    st.markdown("### 📤 Upload X-Ray")
    uploaded_file = st.file_uploader(
        "Choose a dental X-ray image",
        type=['png', 'jpg', 'jpeg'],
        help="Supported formats: PNG, JPG, JPEG"
    )
    
    st.markdown("---")
    st.markdown("""
    <div style='background: #f1f5f9; padding: 1rem; border-radius: 0.5rem;'>
        <h4 style='margin: 0; color: #1e293b;'>📊 Model Info</h4>
        <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #475569;'>
        Architecture: U-Net + ResNet34<br>
        Training: 80 epochs<br>
        Dice Score: 0.343
        </p>
    </div>
    """, unsafe_allow_html=True)

# Load Model (Cached)
@st.cache_resource
def load_model(device):
    possible_models = [
        'checkpoints/dental_model_80_best.pth',
        'checkpoints/dental_model_best.pth',
        'checkpoints/dental_model_last.pth'
    ]
    
    model_path = None
    for path in possible_models:
        if os.path.exists(path):
            model_path = path
            break
            
    if model_path is None:
        st.error("⚠️ No trained model found. Please train the model first using `train.py`")
        return None
    
    with st.spinner('🔄 Loading AI model...'):
        checkpoint = torch.load(model_path, map_location=device)
        model = create_model(arch='Unet', encoder_name='resnet34')
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
    return model

try:
    model = load_model(device)
    if model is not None:
        st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    model = None

def preprocess_image(image):
    transform = A.Compose([
        A.Resize(height=256, width=256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    # Ensure RGB
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    augmented = transform(image=image)
    return augmented['image'].unsqueeze(0)

# Main Interface
if uploaded_file is not None and model is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, 1)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Analysis progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🔍 Preprocessing image...")
    progress_bar.progress(25)
    time.sleep(0.3)
    
    # Predict
    input_tensor = preprocess_image(original_image).to(device)
    
    status_text.text("🧠 Running AI analysis...")
    progress_bar.progress(50)
    
    with torch.no_grad():
        logits = model(input_tensor)
        preds = torch.sigmoid(logits)
        pred_mask = (preds > confidence_threshold).float().cpu().numpy().squeeze()
    
    status_text.text("🎨 Generating visualization...")
    progress_bar.progress(75)
    time.sleep(0.2)
    
    # Resize mask to original size
    pred_mask_resized = cv2.resize(pred_mask, (original_image.shape[1], original_image.shape[0]))
    pred_mask_uint8 = (pred_mask_resized * 255).astype(np.uint8)
    
    # Create Overlay
    overlay = original_image.copy()
    red_layer = np.zeros_like(overlay)
    red_layer[pred_mask_resized > 0.5] = [255, 50, 50]
    overlay = cv2.addWeighted(overlay, 0.7, red_layer, 0.5, 0)
    
    # Draw contours
    contours, _ = cv2.findContours(pred_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), 3)
    
    # Create heatmap
    heatmap = cv2.applyColorMap(pred_mask_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap_overlay = cv2.addWeighted(original_image, 0.6, heatmap_rgb, 0.4, 0)
    
    progress_bar.progress(100)
    status_text.text("✅ Analysis complete!")
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()
    
    # Display Results
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")
    
    # Metrics Row
    caries_area = np.sum(pred_mask_resized > 0.5)
    total_area = pred_mask_resized.size
    percentage = (caries_area / total_area) * 100
    num_regions = len(contours)
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style='margin:0; font-size: 2rem;'>{num_regions}</h3>
            <p style='margin:0.5rem 0 0 0; opacity: 0.9;'>Detected Regions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m2:
        st.markdown(f"""
        <div class="metric-card" style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);'>
            <h3 style='margin:0; font-size: 2rem;'>{percentage:.1f}%</h3>
            <p style='margin:0.5rem 0 0 0; opacity: 0.9;'>Affected Area</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m3:
        confidence_score = float(preds.max().cpu().numpy()) * 100
        st.markdown(f"""
        <div class="metric-card" style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);'>
            <h3 style='margin:0; font-size: 2rem;'>{confidence_score:.1f}%</h3>
            <p style='margin:0.5rem 0 0 0; opacity: 0.9;'>Confidence</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m4:
        severity = "Low" if percentage < 5 else "Medium" if percentage < 15 else "High"
        color = "#10b981" if percentage < 5 else "#f59e0b" if percentage < 15 else "#ef4444"
        st.markdown(f"""
        <div class="metric-card" style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);'>
            <h3 style='margin:0; font-size: 2rem; color: {color};'>{severity}</h3>
            <p style='margin:0.5rem 0 0 0; opacity: 0.9;'>Severity Level</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Image Comparison
    tab1, tab2, tab3, tab4 = st.tabs(["📷 Original", "🎯 Segmentation", "🔥 Heatmap", "📋 Report"])
    
    with tab1:
        st.image(original_image, caption="Original Dental X-Ray", use_column_width=True)
    
    with tab2:
        st.image(overlay, caption=f"Detected Caries - {num_regions} Region(s)", use_column_width=True)
        if num_regions > 0:
            st.markdown(f"""
            <div class="warning-box">
                <strong>⚠️ Detection Alert:</strong> {num_regions} potential caries region(s) identified. 
                Red areas indicate detected caries affecting approximately {percentage:.1f}% of the visible tooth area.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="success-box">
                <strong>✅ No Detection:</strong> No significant caries regions detected in this X-ray. 
                However, professional dental examination is always recommended.
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.image(heatmap_overlay, caption="Probability Heatmap (Red = High, Blue = Low)", use_column_width=True)
        st.info("🌡️ Heatmap shows the probability distribution of caries presence. Warmer colors indicate higher confidence.")
    
    with tab4:
        st.markdown("### 📋 Detailed Analysis Report")
        
        st.markdown(f"""
        <div class="result-box">
            <h4>🔬 Detection Summary</h4>
            <ul>
                <li><strong>Number of detected regions:</strong> {num_regions}</li>
                <li><strong>Total affected area:</strong> {percentage:.2f}% of image</li>
                <li><strong>Maximum confidence:</strong> {confidence_score:.1f}%</li>
                <li><strong>Detection threshold:</strong> {confidence_threshold:.2f}</li>
                <li><strong>Processing device:</strong> {device.upper()}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="result-box">
            <h4>💡 Recommendations</h4>
            <ul>
                <li>This AI analysis is for screening purposes only</li>
                <li>Consult a qualified dentist for professional diagnosis</li>
                <li>Regular dental check-ups are essential for oral health</li>
                <li>Early detection can prevent more serious dental issues</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Download button for report
        if st.button("📥 Download Analysis Report"):
            st.success("Report download feature coming soon!")
    
    # Visualization Chart
    if num_regions > 0:
        st.markdown("---")
        st.markdown("### 📈 Detection Confidence Distribution")
        
        # Create confidence distribution
        confidence_values = preds.cpu().numpy().flatten()
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=confidence_values,
            nbinsx=50,
            marker_color='rgb(99, 102, 241)',
            opacity=0.75,
            name='Confidence Distribution'
        ))
        fig.update_layout(
            title="Pixel-wise Confidence Score Distribution",
            xaxis_title="Confidence Score",
            yaxis_title="Frequency",
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    # Welcome screen
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 3rem 1rem;'>
            <h2 style='color: #1e293b; margin-bottom: 1rem;'>👈 Get Started</h2>
            <p style='color: #64748b; font-size: 1.1rem; line-height: 1.6;'>
                Upload a dental X-ray image using the sidebar to begin AI-powered caries detection.
                Our advanced deep learning model will analyze the image and highlight potential caries regions.
            </p>
            <br>
            <div style='background: #ffffff; padding: 2.5rem; border-radius: 1rem; margin-top: 2rem; border: 2px solid #e2e8f0; box-shadow: 0 4px 6px rgba(0,0,0,0.05);'>
                <h3 style='color: #1e293b; margin-bottom: 1.5rem; font-size: 1.6rem; font-weight: 600;'>✨ Features</h3>
                <div style='text-align: left; display: inline-block;'>
                    <p style='margin: 1rem 0; font-size: 1.05rem; color: #334155; line-height: 1.6;'>🎯 Real-time caries detection</p>
                    <p style='margin: 1rem 0; font-size: 1.05rem; color: #334155; line-height: 1.6;'>🔥 Interactive heatmap visualization</p>
                    <p style='margin: 1rem 0; font-size: 1.05rem; color: #334155; line-height: 1.6;'>📊 Comprehensive analysis metrics</p>
                    <p style='margin: 1rem 0; font-size: 1.05rem; color: #334155; line-height: 1.6;'>⚡ Fast GPU-accelerated processing</p>
                    <p style='margin: 1rem 0; font-size: 1.05rem; color: #334155; line-height: 1.6;'>🎨 Multiple visualization modes</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #94a3b8; padding: 2rem 0;'>
    <p>🦷 <strong>DentalAI</strong> - Powered by Deep Learning | U-Net + ResNet34 Architecture</p>
    <p style='font-size: 0.9rem;'>⚠️ This tool is for educational and research purposes. Always consult a professional dentist.</p>
</div>
""", unsafe_allow_html=True)
