import gradio as gr
from backend.yolo import detect_objects
from backend.openai_client import explain_detection, generate_voice
from backend.face_detection import face_detector
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import json
import requests
import time

def create_detection_chart(detected_objects, face_stats):
    """Create an interactive chart showing detection statistics"""
    
    # Count object types
    object_counts = {}
    for obj in detected_objects:
        object_counts[obj] = object_counts.get(obj, 0) + 1
    
    # Add face detection to counts if faces detected
    if face_stats and face_stats['total_faces'] > 0:
        object_counts['Faces'] = face_stats['total_faces']
        if face_stats['features_detected']['smiles'] > 0:
            object_counts['Smiles'] = face_stats['features_detected']['smiles']
    
    if not object_counts:
        # Create empty chart
        fig = go.Figure()
        fig.add_annotation(
            text="No objects detected",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(
            height=300,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
    else:
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=list(object_counts.keys()),
                y=list(object_counts.values()),
                marker=dict(
                    color=list(range(len(object_counts))),
                    colorscale='Viridis',
                    showscale=False
                ),
                text=list(object_counts.values()),
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Detection Results",
            xaxis_title="Object Type",
            yaxis_title="Count",
            height=300,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(240,240,250,0.5)',
            font=dict(size=12),
            xaxis=dict(gridcolor='white', gridwidth=1),
            yaxis=dict(gridcolor='white', gridwidth=1)
        )
    
    return fig

def create_confidence_pie(face_stats):
    """Create a pie chart for face detection confidence"""
    if not face_stats or face_stats['total_faces'] == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No face data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            height=250,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
    else:
        labels = []
        values = []
        
        if face_stats['features_detected']['smiles'] > 0:
            labels = ['Smiling', 'Not Smiling']
            values = [
                face_stats['features_detected']['smiles'],
                face_stats['total_faces'] - face_stats['features_detected']['smiles']
            ]
        else:
            labels = ['Faces Detected']
            values = [face_stats['total_faces']]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            marker=dict(colors=['#4CAF50', '#FFC107'])
        )])
        
        fig.update_layout(
            title="Face Analysis",
            height=250,
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(size=11)
        )
    
    return fig

def process_image(image, enable_voice=False, enable_face_detection=False):
    # YOLO object detection
    detected_img, detected_objects = detect_objects(image)
    
    # Face detection if enabled
    face_stats = None
    face_analysis = ""
    if enable_face_detection and face_detector:
        detected_img, face_stats = face_detector.detect_faces(detected_img)
        face_analysis = face_detector.analyze_demographics(face_stats)
    
    # Generate explanation
    explanation = explain_detection(detected_objects)
    if face_analysis:
        explanation = f"{explanation}\n\n---\n\n**Face Detection Analysis:**\n{face_analysis}"
    
    # Generate voice if enabled
    audio_file = None
    if enable_voice and explanation != "No objects detected.":
        audio_file = generate_voice(explanation)
    
    # Create statistics
    total_objects = len(detected_objects) + (face_stats['total_faces'] if face_stats else 0)
    stats = f"📊 **Detection Stats**\n"
    stats += f"- Total Objects: {len(detected_objects)}\n"
    if face_stats:
        stats += f"- Faces Found: {face_stats['total_faces']}\n"
        if face_stats['features_detected']['smiles'] > 0:
            stats += f"- Smiles Detected: {face_stats['features_detected']['smiles']}\n"
    stats += f"- Timestamp: {datetime.now().strftime('%I:%M %p')}"
    
    # Create charts
    detection_chart = create_detection_chart(detected_objects, face_stats)
    confidence_pie = create_confidence_pie(face_stats)
    
    return detected_img, explanation, audio_file, stats, detection_chart, confidence_pie

def get_news_headlines():
    """Fetch latest news headlines (mock data for demo)"""
    # In production, this would fetch from a news API
    headlines = [
        "🌍 Global AI Summit Announces New Safety Guidelines",
        "💻 Tech Giants Invest $10B in Computer Vision Research", 
        "🚀 SpaceX Successfully Tests New Rocket Engine",
        "🏥 AI Helps Doctors Detect Cancer Earlier",
        "📱 New Smartphone Features Advanced Face Recognition"
    ]
    return "\n".join(headlines[:3])

custom_css = """
.gradio-container {
    margin-top: 10px !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    font-family: 'Inter', sans-serif;
}

/* Removed non-functional clock and date styles */

/* Collapsible Sidebar Styles */
.sidebar-container {
    transition: all 0.3s ease;
    position: relative;
}

.sidebar-toggle {
    position: absolute;
    top: 10px;
    left: -40px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 10px;
    cursor: pointer;
    border-radius: 5px 0 0 5px;
    z-index: 999;
    font-size: 20px;
    box-shadow: -3px 0 10px rgba(0,0,0,0.2);
}

.sidebar-collapsed {
    margin-left: -350px !important;
}

.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}

.feature-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
}

.launch-button {
    background: black !important;
    color: white !important;
    font-weight: bold !important;
    font-size: 1.2em !important;
    padding: 15px 40px !important;
    border-radius: 8px !important;
    box-shadow: 0 5px 15px rgba(0,0,0,0.3) !important;
    transition: all 0.3s ease !important;
}

.launch-button:hover {
    background: #333 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(0,0,0,0.4) !important;
}

/* News Compass Styles */
.news-compass {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 15px;
    padding: 20px;
    color: white;
    margin-top: 20px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.3);
}

.compass-header {
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 15px;
}

.compass-icon {
    font-size: 30px;
    margin-right: 10px;
    animation: spin 10s linear infinite;
}

@keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

/* Make main content responsive */
.main-content {
    transition: margin-left 0.3s ease;
    width: 100%;
}
"""

# JavaScript for sidebar toggle
js_functions = """
<script>
// Toggle sidebar
let sidebarCollapsed = false;
function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar-container');
    const mainContent = document.querySelector('.main-content');
    const toggleBtn = document.querySelector('.sidebar-toggle');
    
    sidebarCollapsed = !sidebarCollapsed;
    
    if (sidebarCollapsed) {
        sidebar.style.marginLeft = '-350px';
        mainContent.style.marginLeft = '0';
        toggleBtn.innerHTML = '▶';
    } else {
        sidebar.style.marginLeft = '0';
        mainContent.style.marginLeft = '0';
        toggleBtn.innerHTML = '◀';
    }
}
</script>
"""

with gr.Blocks(title="NAVADA - AI Vision", css=custom_css, theme=gr.themes.Soft()) as demo:
    # Add JavaScript for dynamic features
    gr.HTML(js_functions)
    
    with gr.Row():
        gr.HTML("""
        <div class="main-header" style="text-align: center; color: white;">
            <h1 style="font-size: 3em; margin-bottom: 0.5rem;">🚀 NAVADA</h1>
            <h3 style="font-size: 1.5em; font-weight: 400;">AI Computer Vision Application</h3>
            <p style="margin-top: 1rem; font-size: 1.1em; line-height: 1.6;">
                Designed & Developed by <strong>Lee Akpareva</strong><br>
                AI Consultant & Computer Vision Specialist
            </p>
            <p style="margin-top: 1rem; opacity: 0.9;">
                🎓 Created as a demo for Hugging Face to demonstrate<br>
                how to build production-ready ML models in under 15 minutes
            </p>
        </div>
        """)
    
    with gr.Row():
        # Collapsible Sidebar
        with gr.Column(scale=1, elem_id="sidebar", elem_classes="sidebar-container"):
            # Toggle button
            gr.HTML("""
            <button class="sidebar-toggle" onclick="toggleSidebar()">◀</button>
            """)
            
            gr.HTML("""
            <div class="feature-card">
                <h3>✨ Features</h3>
                <ul style="list-style: none; padding-left: 0;">
                    <li>🎯 Real-time object detection using YOLOv8</li>
                    <li>👤 Advanced face detection & analysis</li>
                    <li>😊 Smile & eye detection</li>
                    <li>🤖 AI-powered scene understanding</li>
                    <li>🔊 Text-to-speech narration</li>
                    <li>📊 Interactive detection charts</li>
                    <li>🎨 Beautiful UI with gradient design</li>
                </ul>
            </div>
            """)
            
            gr.HTML("""
            <div class="feature-card">
                <h3>📖 How to Use</h3>
                <ol style="padding-left: 1.2rem;">
                    <li>Upload or drag an image</li>
                    <li>Enable desired features</li>
                    <li>Click "LAUNCH" to analyze</li>
                    <li>View results & statistics</li>
                    <li>Explore interactive charts</li>
                </ol>
            </div>
            """)
            
            gr.HTML("""
            <div class="feature-card">
                <h3>💡 Pro Tips</h3>
                <ul style="padding-left: 1.2rem;">
                    <li>• Enable face detection for portraits</li>
                    <li>• Works with multiple faces</li>
                    <li>• Best with clear, well-lit images</li>
                    <li>• Try group photos for smile analysis</li>
                    <li>• Click ◀ to hide this panel</li>
                </ul>
            </div>
            """)
        
        # Main Content Area
        with gr.Column(scale=3, elem_classes="main-content"):
            # News Compass Section
            gr.HTML(f"""
            <div class="news-compass">
                <div class="compass-header" style="color: white;">
                    <span class="compass-icon">🧭</span>
                    <h3 style="color: white;">News Compass - AI & Tech Headlines</h3>
                </div>
                <div style="font-size: 14px; line-height: 1.8; color: white;">
                    {get_news_headlines()}
                </div>
                <div style="text-align: center; margin-top: 10px; color: rgba(255,255,255,0.8); font-size: 12px;">
                    Live updates every hour • Powered by AI
                </div>
            </div>
            """)
            
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(
                        type="numpy", 
                        label="📷 Upload Image",
                        elem_classes="feature-card"
                    )
                    with gr.Row():
                        enable_voice = gr.Checkbox(
                            label="🔊 Voice Narration", 
                            value=False,
                            info="AI narrates the scene"
                        )
                        enable_face = gr.Checkbox(
                            label="👤 Face Detection", 
                            value=True,
                            info="Detect faces & smiles"
                        )
                
                with gr.Column():
                    output_image = gr.Image(
                        type="numpy", 
                        label="🎯 Detection Results",
                        elem_classes="feature-card"
                    )
                    detection_stats = gr.Markdown(
                        label="Statistics",
                        elem_classes="feature-card"
                    )
            
            # Charts Row
            with gr.Row():
                detection_chart = gr.Plot(
                    label="📊 Detection Chart",
                    elem_classes="feature-card"
                )
                confidence_pie = gr.Plot(
                    label="🥧 Face Analysis",
                    elem_classes="feature-card"
                )
            
            explanation_box = gr.Textbox(
                label="🤖 AI Scene Analysis", 
                lines=8,
                placeholder="AI-generated scene description will appear here...",
                elem_classes="feature-card"
            )
            
            audio_output = gr.Audio(
                label="🎧 Voice Narration",
                visible=True,
                elem_classes="feature-card"
            )
            
            with gr.Row():
                submit_btn = gr.Button(
                    "🚀 LAUNCH", 
                    variant="primary", 
                    size="lg",
                    elem_classes="launch-button"
                )
                clear_btn = gr.Button(
                    "🔄 Clear All", 
                    variant="secondary",
                    size="lg"
                )
    
    with gr.Row():
        gr.HTML("""
        <div style="text-align: center; padding: 2rem; color: #666;">
            <h3>🏆 About This Project</h3>
            <p style="max-width: 800px; margin: 0 auto; line-height: 1.6;">
                NAVADA leverages cutting-edge computer vision, face detection, and natural language processing 
                to provide intelligent image analysis. This demonstration showcases the power of combining YOLOv8 
                object detection with advanced facial recognition and OpenAI's language models.
            </p>
            <p style="margin-top: 1rem;">
                <strong>Technologies:</strong> YOLOv8 | OpenCV | OpenAI GPT-4 | Gradio | Plotly | Python
            </p>
            <p style="margin-top: 1rem; opacity: 0.8;">
                © 2024 Lee Akpareva | AI Innovation Lab
            </p>
        </div>
        """)
    
    submit_btn.click(
        fn=process_image, 
        inputs=[input_image, enable_voice, enable_face], 
        outputs=[output_image, explanation_box, audio_output, detection_stats, detection_chart, confidence_pie]
    )
    
    clear_btn.click(
        fn=lambda: (None, "", None, "", None, None),
        outputs=[output_image, explanation_box, audio_output, detection_stats, detection_chart, confidence_pie]
    )

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)