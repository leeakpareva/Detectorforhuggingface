try:
    import gradio as gr # type: ignore
    from backend.yolo import detect_objects
    from backend.openai_client import explain_detection, generate_voice
    from backend.face_detection import face_detector
    from backend.recognition import recognition_system
    from backend.database import db
    from datetime import datetime
    import plotly.graph_objects as go # type: ignore
    import time
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    print("📦 Please install dependencies: pip install -r requirements.txt")
    raise

def create_detection_chart(detected_objects, face_stats, face_matches=None):
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
    
    # Add recognized faces
    if face_matches:
        known_faces = sum(1 for match in face_matches if match['name'] != 'Unknown')
        if known_faces > 0:
            object_counts['Known Faces'] = known_faces
    
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

def create_confidence_pie(face_stats, face_matches=None):
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
        colors = []
        
        if face_matches:
            known_count = sum(1 for match in face_matches if match['name'] != 'Unknown')
            unknown_count = len(face_matches) - known_count
            
            if known_count > 0:
                labels.append('Known Faces')
                values.append(known_count)
                colors.append('#4CAF50')
            
            if unknown_count > 0:
                labels.append('Unknown Faces')
                values.append(unknown_count)
                colors.append('#FF9800')
        
        if face_stats['features_detected']['smiles'] > 0:
            labels.append('Smiling')
            values.append(face_stats['features_detected']['smiles'])
            colors.append('#FFC107')
        
        if not labels:  # Fallback
            labels = ['Faces Detected']
            values = [face_stats['total_faces']]
            colors = ['#2196F3']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            marker=dict(colors=colors)
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

def process_image(image, enable_voice=False, enable_face_detection=False, enable_recognition=False):
    """Enhanced image processing with recognition capabilities"""
    start_time = time.time()
    
    # YOLO object detection
    detected_img, detected_objects = detect_objects(image)
    
    # Face detection and recognition
    face_stats = None
    face_matches = None
    face_analysis = ""
    
    if enable_face_detection and face_detector:
        if enable_recognition and recognition_system:
            # Use recognition system (includes face detection)
            detected_img, face_matches = recognition_system.recognize_faces(detected_img)
            # Also get face stats for compatibility
            _, face_stats = face_detector.detect_faces(image)
            
            # Create recognition summary
            if face_matches:
                known_faces = [m for m in face_matches if m['name'] != 'Unknown']
                unknown_faces = [m for m in face_matches if m['name'] == 'Unknown']
                
                face_analysis = f"👥 **Face Recognition Results:**\n"
                face_analysis += f"- Total faces detected: {len(face_matches)}\n"
                face_analysis += f"- Known individuals: {len(known_faces)}\n"
                face_analysis += f"- Unknown faces: {len(unknown_faces)}\n\n"
                
                if known_faces:
                    face_analysis += "**Recognized Individuals:**\n"
                    for match in known_faces:
                        face_analysis += f"  • {match['name']} (confidence: {match['similarity']:.2f})\n"
        else:
            # Regular face detection only
            detected_img, face_stats = face_detector.detect_faces(detected_img)
            face_analysis = face_detector.analyze_demographics(face_stats)
    
    # Generate explanation
    explanation = explain_detection(detected_objects)
    if face_analysis:
        explanation = f"{explanation}\n\n---\n\n**Face Analysis:**\n{face_analysis}"
    
    # Add RAG enhancement if recognition is enabled
    if enable_recognition and recognition_system:
        rag_enhancement = recognition_system.enhance_with_rag(detected_objects, face_matches)
        explanation = f"{explanation}\n\n---\n\n{rag_enhancement}"
    
    # Generate voice if enabled
    audio_file = None
    if enable_voice and explanation != "No objects detected.":
        audio_file = generate_voice(explanation)
    
    # Create statistics
    processing_time = time.time() - start_time
    stats = f"📊 **Detection Stats**\n"
    stats += f"- Total Objects: {len(detected_objects)}\n"
    if face_stats:
        stats += f"- Faces Found: {face_stats['total_faces']}\n"
        if face_stats['features_detected']['smiles'] > 0:
            stats += f"- Smiles Detected: {face_stats['features_detected']['smiles']}\n"
    if face_matches:
        known_count = sum(1 for m in face_matches if m['name'] != 'Unknown')
        stats += f"- Known Individuals: {known_count}\n"
    stats += f"- Processing Time: {processing_time:.2f}s\n"
    stats += f"- Timestamp: {datetime.now().strftime('%I:%M %p')}"
    
    # Save to database if recognition is enabled
    if enable_recognition and recognition_system:
        recognition_system.save_session_data(
            image=image,
            detections=detected_objects,
            face_matches=face_matches,
            processing_time=processing_time
        )
    
    # Create charts
    detection_chart = create_detection_chart(detected_objects, face_stats, face_matches)
    confidence_pie = create_confidence_pie(face_stats, face_matches)
    
    return detected_img, explanation, audio_file, stats, detection_chart, confidence_pie

def add_face_to_database(image, name):
    """Add a new face to the recognition database"""
    if not recognition_system or not name.strip():
        return "❌ Recognition system not available or name not provided", ""
    
    success = recognition_system.add_new_face(image, name.strip())
    if success:
        return f"✅ Successfully added '{name}' to face recognition database!", get_database_stats()
    else:
        return f"❌ Failed to add '{name}' to database. Make sure there's a clear face in the image.", ""

def add_object_to_database(image, label, category):
    """Add a custom object to the recognition database"""
    if not recognition_system or not label.strip():
        return "❌ Recognition system not available or label not provided", ""
    
    success = recognition_system.add_custom_object(image, label.strip(), category.strip() or "custom")
    if success:
        return f"✅ Successfully added '{label}' to object recognition database!", get_database_stats()
    else:
        return f"❌ Failed to add '{label}' to database.", ""

def get_database_stats():
    """Get current database statistics"""
    if not db:
        return "Database not available"
    
    stats = db.get_stats()
    if stats:
        return f"""📊 **Database Statistics:**
- Faces: {stats['faces']}
- Custom Objects: {stats['objects']}
- Total Detections: {stats['total_detections']}
- Recent Activity: {stats['recent_detections']} (last 7 days)
- Database Size: {stats['database_size'] / 1024:.1f} KB"""
    else:
        return "Unable to retrieve database statistics"

def get_news_headlines():
    """Fetch latest news headlines (mock data for demo)"""
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
    position: fixed !important;
    top: 150px;
    left: 10px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    padding: 15px 8px !important;
    cursor: pointer !important;
    border-radius: 0 8px 8px 0 !important;
    z-index: 1000 !important;
    font-size: 18px !important;
    box-shadow: 3px 0 15px rgba(0,0,0,0.3) !important;
    width: 40px !important;
    height: 50px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.3s ease !important;
}

.sidebar-toggle:hover {
    background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%) !important;
    transform: translateX(2px) !important;
    box-shadow: 5px 0 20px rgba(0,0,0,0.4) !important;
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

/* Database Panel Styles */
.database-panel {
    background: linear-gradient(135deg, #2e7d32 0%, #388e3c 100%);
    border-radius: 15px;
    padding: 20px;
    color: white;
    margin-top: 20px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.3);
}

.add-button {
    background: #4CAF50 !important;
    color: white !important;
    font-weight: bold !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    box-shadow: 0 3px 10px rgba(76,175,80,0.3) !important;
}

.add-button:hover {
    background: #45a049 !important;
    transform: translateY(-1px) !important;
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
// Toggle sidebar functionality
let sidebarCollapsed = false;

function toggleSidebar() {
    const sidebar = document.querySelector('#sidebar');
    const toggleBtn = document.getElementById('sidebarToggle');
    
    if (!sidebar || !toggleBtn) {
        console.log('Sidebar elements not found, retrying...');
        setTimeout(toggleSidebar, 100);
        return;
    }
    
    sidebarCollapsed = !sidebarCollapsed;
    
    if (sidebarCollapsed) {
        sidebar.style.display = 'none';
        toggleBtn.innerHTML = '▶';
        toggleBtn.style.left = '10px';
        toggleBtn.title = 'Show Sidebar';
    } else {
        sidebar.style.display = 'block';
        toggleBtn.innerHTML = '◀';
        toggleBtn.style.left = '10px';
        toggleBtn.title = 'Hide Sidebar';
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('Page loaded, initializing sidebar toggle...');
    const toggleBtn = document.getElementById('sidebarToggle');
    if (toggleBtn) {
        toggleBtn.title = 'Hide Sidebar';
    }
});

// Backup initialization with delay
setTimeout(function() {
    const toggleBtn = document.getElementById('sidebarToggle');
    if (toggleBtn && !toggleBtn.title) {
        toggleBtn.title = 'Hide Sidebar';
        console.log('Sidebar toggle initialized with delay');
    }
}, 1000);
</script>
"""

with gr.Blocks(title="NAVADA - AI Vision", css=custom_css, theme=gr.themes.Soft()) as demo:
    # Add JavaScript for dynamic features
    gr.HTML(js_functions)
    
    # Add toggle button
    gr.HTML("""
    <button class="sidebar-toggle" onclick="toggleSidebar()" id="sidebarToggle">◀</button>
    """)
    
    with gr.Row():
        gr.HTML("""
        <div class="main-header" style="text-align: center; color: white;">
            <h1 style="font-size: 3em; margin-bottom: 0.5rem;">🚀 NAVADA 2.0</h1>
            <h3 style="font-size: 1.5em; font-weight: 400;">Advanced AI Computer Vision with Recognition & RAG</h3>
            <p style="margin-top: 1rem; font-size: 1.1em; line-height: 1.6;">
                Designed & Developed by <strong>Lee Akpareva</strong><br>
                AI Consultant & Computer Vision Specialist
            </p>
            <p style="margin-top: 1rem; opacity: 0.9;">
                🎓 Enhanced with Custom Recognition Database & RAG Technology<br>
                Now with SQLite storage and intelligent context understanding
            </p>
        </div>
        """)
    
    with gr.Row():
        # Collapsible Sidebar
        with gr.Column(scale=1, elem_id="sidebar", elem_classes="sidebar-container"):
            
            gr.HTML("""
            <div class="feature-card">
                <h3>✨ Enhanced Features</h3>
                <ul style="list-style: none; padding-left: 0;">
                    <li>🎯 Real-time object detection using YOLOv8</li>
                    <li>👤 Advanced face detection & recognition</li>
                    <li>🗄️ Custom face & object database</li>
                    <li>🧠 RAG-enhanced intelligent analysis</li>
                    <li>📸 Photo capture for training data</li>
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
                    <li>Upload an image or use webcam</li>
                    <li>Enable desired detection features</li>
                    <li>Click "LAUNCH" to analyze</li>
                    <li>Add faces/objects to database</li>
                    <li>Enjoy enhanced recognition!</li>
                </ol>
            </div>
            """)
            
            gr.HTML("""
            <div class="feature-card">
                <h3>💡 Pro Tips</h3>
                <ul style="padding-left: 1.2rem;">
                    <li>• Enable recognition for known faces</li>
                    <li>• Add custom objects to improve detection</li>
                    <li>• Use clear, well-lit images</li>
                    <li>• Build your personal recognition database</li>
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
            
            # Database Statistics Panel
            with gr.Row():
                database_stats = gr.Markdown(
                    value=get_database_stats(),
                    elem_classes="database-panel"
                )
            
            # Main Detection Interface
            with gr.Row():
                with gr.Column():
                    # Image input options
                    with gr.Tab("Upload Image"):
                        input_image = gr.Image(
                            type="numpy", 
                            label="📷 Upload Image",
                            elem_classes="feature-card"
                        )
                    
                    with gr.Tab("📸 Capture Photo"):
                        webcam_image = gr.Image(
                            sources=["webcam"],
                            type="numpy",
                            label="📹 Webcam Capture",
                            elem_classes="feature-card"
                        )
                    
                    # Feature controls
                    with gr.Row():
                        enable_voice = gr.Checkbox(
                            label="🔊 Voice Narration", 
                            value=False,
                            info="AI narrates the scene"
                        )
                        enable_face = gr.Checkbox(
                            label="👤 Face Detection", 
                            value=True,
                            info="Detect faces & emotions"
                        )
                        enable_recognition = gr.Checkbox(
                            label="🧠 Smart Recognition",
                            value=True,
                            info="Use custom database & RAG"
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
            
            # Add to Database Section
            with gr.Row():
                with gr.Column():
                    gr.HTML("<h3>➕ Add Face to Database</h3>")
                    face_name_input = gr.Textbox(
                        label="Person's Name",
                        placeholder="Enter person's name..."
                    )
                    add_face_btn = gr.Button(
                        "👤 Add Face",
                        elem_classes="add-button"
                    )
                    face_result = gr.Textbox(
                        label="Result",
                        interactive=False
                    )
                
                with gr.Column():
                    gr.HTML("<h3>🏷️ Add Custom Object</h3>")
                    object_label_input = gr.Textbox(
                        label="Object Label",
                        placeholder="Enter object name..."
                    )
                    object_category_input = gr.Textbox(
                        label="Category",
                        placeholder="Enter category (optional)..."
                    )
                    add_object_btn = gr.Button(
                        "🏷️ Add Object",
                        elem_classes="add-button"
                    )
                    object_result = gr.Textbox(
                        label="Result",
                        interactive=False
                    )
            
            # Charts Row
            with gr.Row():
                detection_chart = gr.Plot(
                    label="📊 Detection Chart",
                    elem_classes="feature-card"
                )
                confidence_pie = gr.Plot(
                    label="🥧 Recognition Analysis",
                    elem_classes="feature-card"
                )
            
            explanation_box = gr.Textbox(
                label="🤖 AI Scene Analysis with RAG Enhancement", 
                lines=10,
                placeholder="AI-generated scene description with intelligent context will appear here...",
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
            <h3>🏆 About NAVADA 2.0</h3>
            <p style="max-width: 800px; margin: 0 auto; line-height: 1.6;">
                NAVADA 2.0 combines cutting-edge computer vision, advanced face recognition, custom object detection,
                and RAG (Retrieval-Augmented Generation) technology. Build your own recognition database, 
                capture training photos, and enjoy intelligent context-aware analysis powered by AI.
            </p>
            <p style="margin-top: 1rem;">
                <strong>Technologies:</strong> YOLOv8 | OpenCV | OpenAI GPT-4 | SQLite | RAG | Gradio | Plotly | Python
            </p>
            <p style="margin-top: 1rem; opacity: 0.8;">
                © 2024 Lee Akpareva | AI Innovation Lab - Enhanced Edition
            </p>
        </div>
        """)
    
    # Event handlers
    def process_with_source_check(upload_img, webcam_img, voice, face_det, recognition):
        """Process image from either upload or webcam"""
        image = upload_img if upload_img is not None else webcam_img
        if image is None:
            return None, "Please upload an image or capture from webcam", None, "", None, None
        return process_image(image, voice, face_det, recognition)
    
    submit_btn.click(
        fn=process_with_source_check, 
        inputs=[input_image, webcam_image, enable_voice, enable_face, enable_recognition], 
        outputs=[output_image, explanation_box, audio_output, detection_stats, detection_chart, confidence_pie]
    )
    
    # Add face functionality
    def add_face_handler(upload_img, webcam_img, name):
        image = upload_img if upload_img is not None else webcam_img
        if image is None:
            return "❌ Please provide an image", ""
        result, stats = add_face_to_database(image, name)
        return result, stats
    
    add_face_btn.click(
        fn=add_face_handler,
        inputs=[input_image, webcam_image, face_name_input],
        outputs=[face_result, database_stats]
    )
    
    # Add object functionality
    def add_object_handler(upload_img, webcam_img, label, category):
        image = upload_img if upload_img is not None else webcam_img
        if image is None:
            return "❌ Please provide an image", ""
        result, stats = add_object_to_database(image, label, category)
        return result, stats
    
    add_object_btn.click(
        fn=add_object_handler,
        inputs=[input_image, webcam_image, object_label_input, object_category_input],
        outputs=[object_result, database_stats]
    )
    
    clear_btn.click(
        fn=lambda: (None, None, "", None, "", "", "", "", None, None),
        outputs=[input_image, webcam_image, explanation_box, audio_output, detection_stats, 
                face_name_input, object_label_input, object_category_input, detection_chart, confidence_pie]
    )

if __name__ == "__main__":
    # Find available port
    import socket
    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]
    
    port = 7860
    try:
        # Test if port is available
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('0.0.0.0', port))
    except OSError:
        port = find_free_port()
        print(f"🔄 Port 7860 in use, using port {port} instead")
    
    print("🚀 Starting NAVADA 2.0...")
    print(f"🌐 Local URL: http://localhost:{port}")
    print(f"🔗 Network URL: http://0.0.0.0:{port}")
    demo.launch(share=False, server_name="0.0.0.0", server_port=port)