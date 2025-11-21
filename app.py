import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Global variables for models
HEALTH_MODEL = None
HEALTH_CLASS_NAMES = None

def load_models():
    """Load health model and class names"""
    global HEALTH_MODEL, HEALTH_CLASS_NAMES
    
    try:
        # Load health model only
        HEALTH_MODEL = tf.keras.models.load_model('health_model.h5')
        
        # Load health class names
        HEALTH_CLASS_NAMES = np.load('health_class_names.npy', allow_pickle=True)
        
        print("✅ Health model loaded successfully!")
        print(f"Health classes: {list(HEALTH_CLASS_NAMES)}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading health model: {e}")
        return False

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Convert to numpy array if it's a PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure image is in RGB format
    if len(image.shape) == 3:
        if image.shape[2] == 4:  # RGBA to RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 1:  # Grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:  # Ensure RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize and normalize
    resize = tf.image.resize(image, (256, 256))
    processed_img = np.expand_dims(resize/255, 0)
    
    return processed_img

def analyze_coral_health(image):
    """Analyze coral image for health assessment only"""
    try:
        # Check if health model is loaded
        if HEALTH_MODEL is None:
            return "❌ Health model not loaded properly. Please check the logs.", None
        
        # Preprocess image
        processed_img = preprocess_image(image)
        
        # Predict health
        health_pred = HEALTH_MODEL.predict(processed_img, verbose=0)
        health_score = health_pred[0][0]
        
        # Determine health prediction
        health_is_positive = health_score > 0.5
        predicted_health = HEALTH_CLASS_NAMES[1] if health_is_positive else HEALTH_CLASS_NAMES[0]
        health_confidence = health_score if health_is_positive else 1 - health_score
        
        # Health status emoji and color
        health_emoji = "💚" if health_is_positive else "💔"
        health_color = "#22c55e" if health_is_positive else "#ef4444"
        
        # Format results
        result = f"""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 15px; color: white; margin-bottom: 20px;">
    <h2 style="margin: 0; text-align: center;">🌊 Coral Health Analysis</h2>
</div>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
    <div style="background: white; padding: 20px; border-radius: 12px; border-left: 5px solid {health_color}; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h3 style="margin: 0 0 15px 0; color: #374151;">{health_emoji} Health Status</h3>
        <p style="font-size: 24px; font-weight: bold; margin: 0; color: {health_color};">{predicted_health.upper()}</p>
    </div>
    
    <div style="background: white; padding: 20px; border-radius: 12px; border-left: 5px solid #3b82f6; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h3 style="margin: 0 0 15px 0; color: #374151;">📊 Confidence</h3>
        <p style="font-size: 24px; font-weight: bold; margin: 0; color: #3b82f6;">{health_confidence:.1%}</p>
    </div>
</div>
<div style="background: #f8fafc; padding: 15px; border-radius: 10px; border: 1px solid #e2e8f0;">
    <h4 style="margin: 0 0 10px 0; color: #64748b;">📈 Detailed Metrics</h4>
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <span style="color: #64748b;">Health Score:</span>
        <span style="font-weight: bold; color: #374151;">{health_score:.3f}</span>
    </div>
</div>
<div style="margin-top: 20px; padding: 15px; background: #ecfdf5; border-radius: 10px; border: 1px solid #a7f3d0;">
    <p style="margin: 0; color: #065f46; font-size: 14px; text-align: center;">
        🤖 Analyzed using deep learning • Focus: Coral health assessment
    </p>
</div>
"""
        
        return result, health_color
        
    except Exception as e:
        return f"❌ Error during health analysis: {str(e)}", "#ef4444"

def clear_analysis():
    """Clear the uploaded image and analysis results"""
    return None, """
<div style="text-align: center; padding: 40px; background: #f8fafc; border-radius: 10px; border: 2px dashed #cbd5e1;">
    <h3 style="color: #64748b; margin-bottom: 10px;">📷 Upload Cleared</h3>
    <p style="color: #94a3b8; margin: 0;">Ready for new coral image upload</p>
</div>
""", "#3b82f6"

# Initialize model at startup
model_loaded = load_models()

# Custom CSS for better styling
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.upload-box {
    border: 2px dashed #cbd5e1 !important;
    border-radius: 12px !important;
    padding: 20px !important;
}
.upload-box:hover {
    border-color: #3b82f6 !important;
}
.analyze-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: bold !important;
}
.analyze-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 15px rgba(102, 126, 234, 0.3) !important;
}
.clear-btn {
    background: #f1f5f9 !important;
    border: 1px solid #cbd5e1 !important;
    color: #64748b !important;
}
.clear-btn:hover {
    background: #e2e8f0 !important;
    transform: translateY(-1px);
}
"""

def create_interface():
    with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Coral Health Analyzer") as demo:
        # Header section
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("""
                <div style="text-align: center;">
                    <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            -webkit-background-clip: text; 
                            -webkit-text-fill-color: transparent;
                            background-clip: text;
                            margin-bottom: 0;">
                        🌊 Coral Health Analyzer
                    </h1>
                    <p style="color: #64748b; font-size: 16px; margin-top: 5px;">
                        Upload a coral image to assess its health status instantly!
                    </p>
                </div>
                """)
        
        # Main content area
        with gr.Row(equal_height=True):
            # Left column - Upload and controls
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 📸 Upload Coral Image")
                    
                    # Image upload with custom styling
                    image_input = gr.Image(
                        type="pil",
                        label="",
                        sources=["upload"],
                        height=300,
                        elem_classes=["upload-box"],
                        show_download_button=False
                    )
                    
                    # Control buttons
                    with gr.Row():
                        analyze_btn = gr.Button(
                            "🔍 Analyze Coral Health", 
                            variant="primary",
                            size="lg",
                            elem_classes=["analyze-btn"]
                        )
                        clear_btn = gr.Button(
                            "🗑️ Clear", 
                            variant="secondary",
                            size="lg",
                            elem_classes=["clear-btn"]
                        )
                
                # Instructions
                with gr.Accordion("ℹ️ How to use", open=False):
                    gr.Markdown("""
                    **Simple Steps:**
                    1. 📷 **Upload** - Click the upload area or drag & drop a coral image
                    2. 🔍 **Analyze** - Click the 'Analyze Coral Health' button
                    3. 📊 **Review** - See the health assessment results
                    4. 🗑️ **Clear** - Remove the current image to analyze a new one
                    
                    **Supported formats:** JPG, PNG, WebP
                    **Best results:** Clear, well-lit coral photos
                    """)
            
            # Right column - Results
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Analysis Results")
                
                # Results display with initial state
                output_html = gr.HTML(
                    value="""
                    <div style="text-align: center; padding: 40px; background: #f8fafc; border-radius: 10px; border: 2px dashed #cbd5e1;">
                        <h3 style="color: #64748b; margin-bottom: 10px;">🔄 Ready for Analysis</h3>
                        <p style="color: #94a3b8; margin: 0;">Upload a coral image and click 'Analyze' to see results</p>
                    </div>
                    """
                )
                
                # Hidden component to track health color
                health_color_tracker = gr.Textbox(visible=False, value="#3b82f6")
        
        # Features section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🌟 Features")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div style="text-align: center; padding: 20px; background: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                            <div style="font-size: 24px; margin-bottom: 10px;">⚡</div>
                            <h4 style="margin: 0 0 10px 0; color: #374151;">Instant Analysis</h4>
                            <p style="color: #64748b; margin: 0; font-size: 14px;">Get health results in seconds using AI</p>
                        </div>
                        """)
                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div style="text-align: center; padding: 20px; background: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                            <div style="font-size: 24px; margin-bottom: 10px;">🎯</div>
                            <h4 style="margin: 0 0 10px 0; color: #374151;">Accurate Assessment</h4>
                            <p style="color: #64748b; margin: 0; font-size: 14px;">Trained on real coral reef data</p>
                        </div>
                        """)
                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div style="text-align: center; padding: 20px; background: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                            <div style="font-size: 24px; margin-bottom: 10px;">💾</div>
                            <h4 style="margin: 0 0 10px 0; color: #374151;">Easy Management</h4>
                            <p style="color: #64748b; margin: 0; font-size: 14px;">Simple upload and clear workflow</p>
                        </div>
                        """)
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 20px; background: #f8fafc; border-radius: 10px;">
            <p style="color: #64748b; margin: 0;">
                Built with ❤️ using TensorFlow & Gradio • 
                <span style="color: #3b82f6;">Helping protect coral reefs through AI</span>
            </p>
        </div>
        """)
        
        # Event handlers
        analyze_btn.click(
            fn=analyze_coral_health,
            inputs=[image_input],
            outputs=[output_html, health_color_tracker]
        )
        
        clear_btn.click(
            fn=clear_analysis,
            inputs=[],
            outputs=[image_input, output_html, health_color_tracker]
        )
    
    return demo

# Create and launch interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

