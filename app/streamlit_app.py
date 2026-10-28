import os
import sys
from pathlib import Path
from datetime import datetime
import tempfile
import time
from PIL import Image
import cv2
import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go
from fpdf import FPDF
import io
import base64
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import plotly.io as pio

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Create videos directory if it doesn't exist
VIDEOS_DIR = os.path.join(project_root, 'videos')
Path(VIDEOS_DIR).mkdir(parents=True, exist_ok=True)

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from models.emotion_model import (
    EmotionCNN,
    EmotionCNNModern,
    EmotionCNNRegularized,
    load_model
)
import torchvision.transforms as transforms
import time
from models.detect_emotions import SalesCallAnalyzer, process_video
from config import MODEL_TYPE_PROD, MODEL_PATH, CONFIDENCE_THRESHOLD, ANALYSIS_INTERVAL
import plotly.graph_objects as go

EMOTION_LABELS_KK = ['Ашулы', 'Жиіркену', 'Қорқыныш', 'Қуаныш', 'Қайғы', 'Таңданыс', 'Бейтарап']

def main():
    st.set_page_config(
        page_title="Студенттер эмоциясын талдау жүйесі",
        page_icon="😊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .main {
            background-color: #f4f6fb;
            padding: 2rem;
        }

        .hero-banner {
            background: linear-gradient(135deg, #1a237e 0%, #3949ab 50%, #1e88e5 100%);
            border-radius: 16px;
            padding: 2.5rem 2rem;
            margin-bottom: 2rem;
            color: white;
            box-shadow: 0 8px 32px rgba(30, 136, 229, 0.25);
        }
        .hero-banner h1 {
            font-size: 2rem;
            font-weight: 700;
            margin: 0 0 0.5rem 0;
            color: white;
        }
        .hero-banner p {
            font-size: 1rem;
            opacity: 0.88;
            margin: 0;
            color: white;
        }
        .hero-badge {
            display: inline-block;
            background: rgba(255,255,255,0.2);
            border: 1px solid rgba(255,255,255,0.35);
            border-radius: 20px;
            padding: 4px 14px;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            margin-bottom: 1rem;
            color: white;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: #e8eaf6;
            border-radius: 12px;
            padding: 6px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 3rem;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 8px;
            padding: 0 1.5rem;
            font-weight: 500;
            color: #3949ab;
            border: none;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #3949ab, #1e88e5) !important;
            color: white !important;
            box-shadow: 0 4px 12px rgba(57, 73, 171, 0.35);
        }

        [data-testid="metric-container"] {
            background: white;
            border-radius: 12px;
            padding: 1.2rem 1rem;
            box-shadow: 0 2px 12px rgba(0,0,0,0.07);
            border-left: 4px solid #3949ab;
        }

        .stButton > button {
            background: linear-gradient(135deg, #3949ab, #1e88e5);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            box-shadow: 0 4px 14px rgba(57, 73, 171, 0.35);
            transition: all 0.2s ease;
            width: 100%;
        }
        .stButton > button:hover {
            box-shadow: 0 6px 20px rgba(57, 73, 171, 0.5);
            transform: translateY(-1px);
        }

        [data-testid="stFileUploader"] {
            background: white;
            border: 2px dashed #c5cae9;
            border-radius: 12px;
            padding: 1rem;
        }

        h2, h3 {
            color: #1a237e;
        }

        .streamlit-expanderHeader {
            background: #e8eaf6;
            border-radius: 10px;
            font-weight: 600;
            color: #1a237e;
        }

        .stSuccess, .stInfo, .stWarning, .stError {
            border-radius: 10px;
            font-weight: 500;
        }

        hr {
            border-color: #c5cae9;
            margin: 1.5rem 0;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a237e 0%, #283593 100%);
        }
        section[data-testid="stSidebar"] * {
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="hero-banner">
            <div class="hero-badge">AI · Deep Learning · Computer Vision</div>
            <h1>Студенттер эмоциясын талдау жүйесі</h1>
            <p>Жүйе онлайн-сабақтар кезінде студенттердің эмоционалдық күйін талдайды.<br>
            Дәріс жазбасын немесе фотосуретті жүктеңіз — ұсыныстармен толық есеп алыңыз.</p>
        </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📹 Дәрісті талдау", "🖼️ Студент фотосын талдау"])

    with tab1:
        st.header("Дәріс жазбасын талдау")
        st.markdown("""
            Студенттің бүкіл сабақ бойы эмоцияларын талдау үшін онлайн-сабақтың жазбасын жүктеңіз.
            Жүйе белсенділік деңгейін, назар жоғалту сәттерін анықтап, ұсыныстар береді.
        """)

        uploaded_file = st.file_uploader("Дәріс жазбасын таңдаңыз", type=['mp4', 'avi', 'mov'])

        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name

            if st.button("Дәрісті талдау", type="primary"):
                with st.spinner("Дәріс жазбасы талдануда..."):
                    try:
                        results = process_video(video_path)

                        if results is None:
                            st.error("Жазбаны өңдеуде қате орын алды")
                            return

                        st.success("Дәріс талдауы аяқталды!")

                        st.subheader("Талдау нәтижелері")

                        total_frames = results['total_frames']
                        emotion_counts = results['emotion_counts']
                        face_detection_count = results['face_detection_count']
                        face_coverage_distribution = results['face_coverage_distribution']
                        emotion_percentages = {emotion: (count/total_frames)*100 for emotion, count in emotion_counts.items()}
                        face_detection_percentage = (face_detection_count/total_frames)*100

                        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                        with metrics_col1:
                            st.metric(
                                "Оқуға белсенділік",
                                f"{results['avg_engagement']:.2f}",
                                f"{'↑ өсуде' if results['engagement_trend'] > 0 else '↓ төмендеуде'}"
                            )
                        with metrics_col2:
                            st.metric(
                                "Назар тұрақтылығы",
                                f"{results['emotion_stability']:.2f}",
                                "Жоғары — назар тұрақты"
                            )
                        with metrics_col3:
                            st.metric(
                                "Студент болуы",
                                f"{face_detection_percentage:.1f}%",
                                f"{face_detection_count} кадр"
                            )

                        st.subheader("Сабақ кезіндегі студент күйі")
                        total_emotions = sum(emotion_counts.values()) or 1
                        pct_engaged = (emotion_counts.get('Қуаныш', 0) + emotion_counts.get('Таңданыс', 0)) / total_emotions * 100
                        pct_neutral = emotion_counts.get('Бейтарап', 0) / total_emotions * 100
                        pct_confused = (emotion_counts.get('Қайғы', 0) + emotion_counts.get('Қорқыныш', 0)) / total_emotions * 100
                        pct_frustrated = (emotion_counts.get('Ашулы', 0) + emotion_counts.get('Жиіркену', 0)) / total_emotions * 100

                        interp_col1, interp_col2, interp_col3, interp_col4 = st.columns(4)
                        with interp_col1:
                            st.metric("Белсенді қатысуда 😊", f"{pct_engaged:.1f}%", "Қуаныш + Таңданыс")
                        with interp_col2:
                            st.metric("Зейінді 😐", f"{pct_neutral:.1f}%", "Бейтарап")
                        with interp_col3:
                            st.metric("Қиыншылық сезінуде 😟", f"{pct_confused:.1f}%", "Қайғы + Қорқыныш")
                        with interp_col4:
                            st.metric("Шаршады / ашулы 😠", f"{pct_frustrated:.1f}%", "Ашулы + Жиіркену")

                        if pct_engaged >= 40:
                            st.success("Студент оқу үдерісіне белсенді қатысуда. Материал жақсы қабылдануда.")
                        elif pct_neutral >= 50:
                            st.info("Студент зейінді, бірақ айқын қызығушылықсыз. Интерактивті элементтер қосу ұсынылады.")
                        elif pct_confused >= 30:
                            st.warning("Студент қиыншылық сезінуде. Материал берілісін жеңілдету немесе сұрақтарға үзіліс жасау ұсынылады.")
                        elif pct_frustrated >= 20:
                            st.error("Студент ашулы немесе шаршаған. Үзіліс жасау ұсынылады.")

                        main_col1, main_col2 = st.columns(2)

                        with main_col1:
                            fig_emotions = go.Figure()
                            fig_emotions.add_trace(go.Bar(
                                y=list(emotion_counts.keys()),
                                x=list(emotion_counts.values()),
                                orientation='h',
                                text=[f'{v} кадр\n({emotion_percentages[k]:.1f}%)' for k, v in emotion_counts.items()],
                                textposition='auto',
                                textfont=dict(size=14),
                                marker=dict(
                                    color=['#1e88e5', '#1976d2', '#1565c0', '#0d47a1', '#42a5f5', '#90caf9', '#bbdefb']
                                ),
                                name='Эмоциялар жиілігі'
                            ))
                            fig_emotions.update_layout(
                                title='Эмоциялар үлестірімі',
                                xaxis_title='Кадрлар саны',
                                showlegend=False,
                                height=400,
                                font=dict(size=14),
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                margin=dict(l=20, r=20, t=40, b=20)
                            )
                            st.plotly_chart(fig_emotions, use_container_width=True)

                        with main_col2:
                            fig_faces = go.Figure()
                            fig_faces.add_trace(go.Pie(
                                labels=['Беті бар кадрлар', 'Беті жоқ кадрлар'],
                                values=[face_detection_count, total_frames - face_detection_count],
                                hole=.4,
                                marker=dict(colors=['#1e88e5', '#90caf9']),
                                textinfo='label+percent',
                                textfont_size=14
                            ))
                            fig_faces.update_layout(
                                title='Беттерді анықтау',
                                showlegend=False,
                                height=400,
                                font=dict(size=14),
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                margin=dict(l=20, r=20, t=40, b=20)
                            )
                            st.plotly_chart(fig_faces, use_container_width=True)

                        trend_col1, trend_col2 = st.columns(2)

                        with trend_col1:
                            transitions = results['emotion_transitions']
                            top_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:5]

                            fig_transitions = go.Figure()
                            fig_transitions.add_trace(go.Bar(
                                y=[t[0] for t in top_transitions],
                                x=[t[1] for t in top_transitions],
                                orientation='h',
                                text=[f'{t[1]} рет' for t in top_transitions],
                                textposition='auto',
                                textfont=dict(size=14),
                                marker=dict(
                                    color=['#1e88e5', '#1976d2', '#1565c0', '#0d47a1', '#42a5f5']
                                )
                            ))
                            fig_transitions.update_layout(
                                title='Эмоциялар арасындағы үздік-5 өтпелі',
                                xaxis_title='Өтпелілер саны',
                                showlegend=False,
                                height=300,
                                font=dict(size=14),
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                margin=dict(l=20, r=20, t=40, b=20)
                            )
                            st.plotly_chart(fig_transitions, use_container_width=True)

                        with trend_col2:
                            fig_coverage = go.Figure()
                            fig_coverage.add_trace(go.Bar(
                                y=list(face_coverage_distribution.keys()),
                                x=list(face_coverage_distribution.values()),
                                orientation='h',
                                text=[f'{v} кадр' for v in face_coverage_distribution.values()],
                                textposition='auto',
                                textfont=dict(size=14),
                                marker=dict(color=['#1e88e5', '#42a5f5', '#90caf9'])
                            ))
                            fig_coverage.update_layout(
                                title='Кадрды беттермен жабу',
                                xaxis_title='Кадрлар саны',
                                showlegend=False,
                                height=300,
                                font=dict(size=14),
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                margin=dict(l=20, r=20, t=40, b=20)
                            )
                            st.plotly_chart(fig_coverage, use_container_width=True)

                        fig_engagement = go.Figure()
                        fig_engagement.add_trace(go.Scatter(
                            x=results['time_segments'],
                            y=results['engagement_scores'],
                            mode='lines',
                            line=dict(color='#1e88e5', width=2),
                            name='Белсенділік',
                            fill='tozeroy',
                            fillcolor='rgba(30, 136, 229, 0.1)',
                            hovertemplate='Уақыт: %{x:.1f} сек<br>Белсенділік: %{y:.2f}<extra></extra>'
                        ))
                        fig_engagement.update_layout(
                            title='Белсенділік динамикасы',
                            xaxis_title='Уақыт (секунд)',
                            yaxis_title='Белсенділік деңгейі',
                            showlegend=False,
                            height=300,
                            font=dict(size=14),
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            margin=dict(l=20, r=20, t=40, b=20),
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig_engagement, use_container_width=True)

                        with st.expander("Сабақ қорытындысы", expanded=True):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Сабақ ұзақтығы", f"{results['duration']:.2f} сек")
                            with col2:
                                st.metric("Барлық кадрлар", f"{total_frames:,}")
                            with col3:
                                st.metric("Студент бар кадрлар", f"{face_detection_count:,}")

                        st.subheader("Өңделген жазба")
                        if os.path.exists(results['output_path']):
                            video_file = open(results['output_path'], 'rb')
                            video_bytes = video_file.read()
                            st.video(video_bytes)

                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(
                                    label="Өңделген жазбаны жүктеп алу",
                                    data=video_bytes,
                                    file_name=os.path.basename(results['output_path']),
                                    mime="video/mp4"
                                )
                            with col2:
                                st.download_button(
                                    label="Сабақ бойынша есепті жүктеп алу (PDF)",
                                    data=results['pdf_report'].getvalue(),
                                    file_name=f"student_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                    mime="application/pdf"
                                )
                        else:
                            st.error(f"Өңделген бейнені табу мүмкін болмады: {results['output_path']}")

                    except Exception as e:
                        st.error(f"Бейнені өңдеуде қате: {str(e)}")
                    finally:
                        try:
                            if os.path.exists(video_path):
                                os.unlink(video_path)
                        except Exception as e:
                            st.warning(f"Уақытша файлды жою мүмкін болмады: {str(e)}")

    with tab2:
        st.header("Студент фотосын талдау")
        st.markdown("""
            Студенттің эмоционалдық күйін жылдам талдау үшін фотосуретін жүктеңіз.
            Жүйе қазіргі эмоцияны анықтап, оқуға белсенділік деңгейін бағалайды.
        """)

        uploaded_file = st.file_uploader("Студент фотосын таңдаңыз...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            if st.button("Фотосуретті талдау", type="primary"):
                analyze_image(uploaded_file)
            else:
                st.info("Талдауды бастау үшін 'Фотосуретті талдау' батырмасын басыңыз")

def display_results():
    """Талдау нәтижелерін көрсету."""
    st.header("Талдау нәтижелері")

    total_frames = st.session_state.results['total_frames']
    emotion_counts = st.session_state.results['emotion_counts']
    face_detection_count = st.session_state.results['face_detection_count']

    emotion_percentages = {emotion: (count/total_frames)*100 for emotion, count in emotion_counts.items()}
    face_detection_percentage = (face_detection_count/total_frames)*100

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Эмоциялар жиілігі")
        fig1 = go.Figure(data=[
            go.Bar(
                y=list(emotion_counts.keys()),
                x=list(emotion_counts.values()),
                orientation='h',
                text=[f'{v} кадр\n({emotion_percentages[k]:.1f}%)' for k, v in emotion_counts.items()],
                textposition='auto',
                textfont=dict(size=14),
                marker=dict(
                    color=['#1e88e5', '#1976d2', '#1565c0', '#0d47a1', '#42a5f5', '#90caf9', '#bbdefb']
                )
            )
        ])
        fig1.update_layout(
            xaxis_title='Кадрлар саны',
            showlegend=False,
            height=400,
            font=dict(size=14),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Беттерді анықтау")
        fig2 = go.Figure(data=[
            go.Bar(
                y=['Беті бар кадрлар', 'Беті жоқ кадрлар'],
                x=[face_detection_count, total_frames - face_detection_count],
                orientation='h',
                text=[f'{face_detection_count} кадр\n({face_detection_percentage:.1f}%)',
                      f'{total_frames - face_detection_count} кадр\n({100-face_detection_percentage:.1f}%)'],
                textposition='auto',
                textfont=dict(size=14),
                marker=dict(color=['#1e88e5', '#90caf9'])
            )
        ])
        fig2.update_layout(
            xaxis_title='Кадрлар саны',
            showlegend=False,
            height=200,
            font=dict(size=14),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        st.subheader("Кадрды беттермен жабу")
        coverage_data = st.session_state.results['face_coverage_distribution']
        fig3 = go.Figure(data=[
            go.Bar(
                y=list(coverage_data.keys()),
                x=list(coverage_data.values()),
                orientation='h',
                text=[f'{v} кадр' for v in coverage_data.values()],
                textposition='auto',
                textfont=dict(size=14),
                marker=dict(color=['#1e88e5', '#42a5f5', '#90caf9'])
            )
        ])
        fig3.update_layout(
            xaxis_title='Кадрлар саны',
            showlegend=False,
            height=200,
            font=dict(size=14),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.header("Өңделген бейне")
    video_file = open(st.session_state.results['output_path'], 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)

    st.download_button(
        label="Өңделген бейнені жүктеп алу",
        data=video_bytes,
        file_name=os.path.basename(st.session_state.results['output_path']),
        mime="video/mp4"
    )

def analyze_image(uploaded_file):
    """Жүктелген суреттегі эмоцияларды талдау."""
    analyzer = SalesCallAnalyzer()
    st.session_state.analyzer = analyzer

    image = Image.open(uploaded_file)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    faces, gray = analyzer.detect_faces(image)

    if len(faces) == 0:
        st.warning("Суретте бет анықталмады")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        image_with_boxes = image_rgb.copy()
        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(image_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image_with_boxes, f"Bet {i+1}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        st.image(image_with_boxes, caption="Анықталған беттері бар сурет", width=600)

    with col2:
        face_options = [f"Бет {i+1}" for i in range(len(faces))]
        selected_face = st.selectbox("Талдау үшін бетті таңдаңыз:", face_options)
        selected_idx = int(selected_face.split()[1]) - 1

        x, y, w, h = faces[selected_idx]
        face_roi = gray[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
        st.image(face_rgb, caption=f"Бет {selected_idx+1}", width=300)

        face_tensor = analyzer.preprocess_face(face_roi)
        face_tensor = face_tensor.to(analyzer.device)

        with torch.no_grad():
            outputs = analyzer.model(face_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probabilities, 1)
            emotion_idx = predicted.item()
            confidence = probabilities[0][emotion_idx].item()

            emotion = EMOTION_LABELS_KK[emotion_idx]

            st.write(f"Анықталған эмоция: {emotion}")
            st.write(f"Сенімділік: {confidence:.2f}")

            probs = probabilities[0].cpu().numpy()
            fig = go.Figure(data=[
                go.Bar(
                    y=EMOTION_LABELS_KK,
                    x=probs,
                    orientation='h',
                    text=[f'{p:.2f}' for p in probs],
                    textposition='auto',
                    textfont=dict(size=16),
                    marker=dict(
                        color=['#e74c3c', '#9b59b6', '#3498db', '#2ecc71', '#f1c40f', '#e67e22', '#1abc9c']
                    )
                )
            ])
            fig.update_layout(
                title='Эмоциялар ықтималдылығы',
                xaxis_title='Ықтималдылық',
                yaxis_title='Эмоция',
                xaxis_range=[0, 1],
                height=400,
                showlegend=False,
                font=dict(size=16),
                yaxis=dict(tickfont=dict(size=16)),
                xaxis=dict(tickfont=dict(size=16)),
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            st.plotly_chart(fig, use_container_width=True)

def generate_pdf_report(results, video_path, emotion_screenshots=None):
    """Талдау нәтижелері бойынша PDF есеп жасау."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    story.append(Paragraph("Studenttyn emotsiyalyq taldau eseby", title_style))
    story.append(Spacer(1, 20))

    story.append(Paragraph(f"Jasalgan kuni: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Paragraph(f"Sabaq zhazbasy: {os.path.basename(video_path)}", styles['Normal']))
    story.append(Spacer(1, 20))

    story.append(Paragraph("Negizgi korsеtkіshter", styles['Heading2']))
    metrics_data = [
        ["Korsеtkіsh", "Мan"],
        ["Oquga belsendilik", f"{results['avg_engagement']:.2f}"],
        ["Nazar turaqtylyghy", f"{results['emotion_stability']:.2f}"],
        ["Student boluy", f"{(results['face_detection_count']/results['total_frames'])*100:.1f}%"],
        ["Sabaq uzaqtyghy", f"{results['duration']:.2f} sek"],
        ["Barlyq kadrlar", f"{results['total_frames']:,}"]
    ]
    metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e88e5')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#e3f2fd')),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#1e88e5'))
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 20))

    story.append(Paragraph("Emotsiyalar ulestrimi", styles['Heading2']))
    fig_emotions = go.Figure()
    fig_emotions.add_trace(go.Bar(
        y=list(results['emotion_counts'].keys()),
        x=list(results['emotion_counts'].values()),
        orientation='h',
        text=[f'{v} kadr\n({(v/results["total_frames"])*100:.1f}%)' for k, v in results['emotion_counts'].items()],
        textposition='auto',
        textfont=dict(size=14),
        marker=dict(
            color=['#1e88e5', '#1976d2', '#1565c0', '#0d47a1', '#42a5f5', '#90caf9', '#bbdefb']
        )
    ))
    fig_emotions.update_layout(
        title='Emotsiyalar ulestrimi',
        xaxis_title='Kadrlar sany',
        showlegend=False,
        height=400,
        font=dict(size=14),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    emotion_chart_bytes = pio.to_image(fig_emotions, format='png', width=680, height=510)
    emotion_chart_buffer = io.BytesIO(emotion_chart_bytes)
    story.append(RLImage(emotion_chart_buffer, width=5.1*inch, height=3.825*inch))
    story.append(Spacer(1, 20))

    story.append(Paragraph("Kadrdagy student boluy", styles['Heading2']))
    fig_faces = go.Figure()
    fig_faces.add_trace(go.Pie(
        labels=['Beti bar kadrlar', 'Beti zhok kadrlar'],
        values=[results['face_detection_count'], results['total_frames'] - results['face_detection_count']],
        hole=.4,
        marker=dict(colors=['#1e88e5', '#90caf9']),
        textinfo='label+percent',
        textfont_size=14
    ))
    fig_faces.update_layout(
        title='Betterdi anyqtau',
        showlegend=False,
        height=400,
        font=dict(size=14),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    faces_chart_bytes = pio.to_image(fig_faces, format='png', width=680, height=510)
    faces_chart_buffer = io.BytesIO(faces_chart_bytes)
    story.append(RLImage(faces_chart_buffer, width=5.1*inch, height=3.825*inch))
    story.append(Spacer(1, 20))

    story.append(Paragraph("Emotsiyalar arasy top otpeliler", styles['Heading2']))
    transitions = sorted(results['emotion_transitions'].items(), key=lambda x: x[1], reverse=True)[:5]
    fig_transitions = go.Figure()
    fig_transitions.add_trace(go.Bar(
        y=[t[0] for t in transitions],
        x=[t[1] for t in transitions],
        orientation='h',
        text=[f'{t[1]} ret' for t in transitions],
        textposition='auto',
        textfont=dict(size=14),
        marker=dict(color=['#1e88e5', '#1976d2', '#1565c0', '#0d47a1', '#42a5f5'])
    ))
    fig_transitions.update_layout(
        title='Top 5 emotsiya otpelileri',
        xaxis_title='Otpeliler sany',
        showlegend=False,
        height=300,
        font=dict(size=14),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    transitions_chart_bytes = pio.to_image(fig_transitions, format='png', width=680, height=340)
    transitions_chart_buffer = io.BytesIO(transitions_chart_bytes)
    story.append(RLImage(transitions_chart_buffer, width=5.1*inch, height=2.55*inch))
    story.append(Spacer(1, 20))

    story.append(Paragraph("Kadrdyn betipen zhabyluy", styles['Heading2']))
    fig_coverage = go.Figure()
    fig_coverage.add_trace(go.Bar(
        y=list(results['face_coverage_distribution'].keys()),
        x=list(results['face_coverage_distribution'].values()),
        orientation='h',
        text=[f'{v} kadr' for v in results['face_coverage_distribution'].values()],
        textposition='auto',
        textfont=dict(size=14),
        marker=dict(color=['#1e88e5', '#42a5f5', '#90caf9'])
    ))
    fig_coverage.update_layout(
        title='Kadrdyn betipen zhabyluy',
        xaxis_title='Kadrlar sany',
        showlegend=False,
        height=300,
        font=dict(size=14),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    coverage_chart_bytes = pio.to_image(fig_coverage, format='png', width=680, height=340)
    coverage_chart_buffer = io.BytesIO(coverage_chart_bytes)
    story.append(RLImage(coverage_chart_buffer, width=5.1*inch, height=2.55*inch))
    story.append(Spacer(1, 20))

    story.append(Paragraph("Oqu belsendiliginin dinamikasy", styles['Heading2']))
    fig_engagement = go.Figure()
    fig_engagement.add_trace(go.Scatter(
        x=results['time_segments'],
        y=results['engagement_scores'],
        mode='lines',
        line=dict(color='#1e88e5', width=2),
        name='Belsendilik',
        fill='tozeroy',
        fillcolor='rgba(30, 136, 229, 0.1)',
        hovertemplate='Uaqyt: %{x:.1f} sek<br>Belsendilik: %{y:.2f}<extra></extra>'
    ))
    fig_engagement.update_layout(
        title='Belsendilik dinamikasy',
        xaxis_title='Uaqyt (sekund)',
        yaxis_title='Belsendilik dengei',
        showlegend=False,
        height=300,
        font=dict(size=14),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode='x unified'
    )
    engagement_chart_bytes = pio.to_image(fig_engagement, format='png', width=680, height=340)
    engagement_chart_buffer = io.BytesIO(engagement_chart_bytes)
    story.append(RLImage(engagement_chart_buffer, width=5.1*inch, height=2.55*inch))
    story.append(Spacer(1, 20))

    story.append(Paragraph("Emotsiyalar ulestrimi (keste)", styles['Heading2']))
    emotion_data = [
        ["Emotsiya", "Sany", "Payyzy"],
        *[[emotion, str(count), f"{(count/results['total_frames'])*100:.1f}%"]
          for emotion, count in results['emotion_counts'].items()]
    ]
    emotion_table = Table(emotion_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
    emotion_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e88e5')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#e3f2fd')),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#1e88e5'))
    ]))
    story.append(emotion_table)
    story.append(Spacer(1, 20))

    story.append(Paragraph("Emotsiyalar arasy otpeliler (keste)", styles['Heading2']))
    transition_data = [
        ["Otpeli", "Sany"],
        *[[transition, str(count)] for transition, count in transitions]
    ]
    transition_table = Table(transition_data, colWidths=[3*inch, 2*inch])
    transition_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e88e5')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#e3f2fd')),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#1e88e5'))
    ]))
    story.append(transition_table)
    story.append(Spacer(1, 20))

    if emotion_screenshots:
        story.append(Paragraph("Emotsiyaly kadrlar misaldary", styles['Heading2']))
        for emotion, screenshot in emotion_screenshots.items():
            if screenshot is not None:
                img = Image.fromarray(cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB))
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                story.append(Paragraph(f"Emotsiya: {emotion}", styles['Heading3']))
                story.append(RLImage(img_buffer, width=3.4*inch, height=2.55*inch))
                story.append(Spacer(1, 20))

    doc.build(story)
    buffer.seek(0)
    return buffer

def process_video(video_path):
    """Бейне файлды эмоциялық талдау үшін өңдеу."""
    analyzer = SalesCallAnalyzer()
    st.session_state.analyzer = analyzer

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Қате: бейне файлды ашу мүмкін болмады")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_area = width * height
    duration = total_frames / fps

    output_filename = f"output_{os.path.basename(video_path)}"
    output_path = os.path.join(VIDEOS_DIR, output_filename)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        st.error(f"Қате: шығыс бейне файлды жасау мүмкін болмады {output_path}")
        return

    progress_bar = st.progress(0)
    status_text = st.empty()

    processed_frames = 0
    emotion_counts = {
        'Ашулы': 0, 'Жиіркену': 0, 'Қорқыныш': 0, 'Қуаныш': 0,
        'Қайғы': 0, 'Таңданыс': 0, 'Бейтарап': 0
    }
    face_detection_count = 0
    face_coverage_distribution = {
        'Аз (<25%)': 0,
        'Орташа (25-75%)': 0,
        'Көп (>75%)': 0
    }

    emotion_sequence = []
    engagement_scores = []
    time_segments = []
    emotion_transitions = {}
    last_emotion = None

    emotion_screenshots = {
        'Ашулы': None, 'Жиіркену': None, 'Қорқыныш': None, 'Қуаныш': None,
        'Қайғы': None, 'Таңданыс': None, 'Бейтарап': None
    }

    engagement_weights = {
        'Қуаныш': 1.0,
        'Таңданыс': 0.9,
        'Бейтарап': 0.7,
        'Қайғы': 0.3,
        'Қорқыныш': 0.2,
        'Ашулы': 0.1,
        'Жиіркену': 0.1
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        progress_percentage = (processed_frames / total_frames) * 100
        current_time = processed_frames / fps

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = analyzer.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20)
        )

        frame_face_area = sum(w * h for (x, y, w, h) in faces)
        if len(faces) > 0:
            face_detection_count += 1
            face_coverage = (frame_face_area / frame_area) * 100

            if face_coverage < 25:
                face_coverage_distribution['Аз (<25%)'] += 1
            elif face_coverage < 75:
                face_coverage_distribution['Орташа (25-75%)'] += 1
            else:
                face_coverage_distribution['Көп (>75%)'] += 1

        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            face_roi = gray[y:y+h, x:x+w]
            face_tensor = analyzer.preprocess_face(face_roi)
            face_tensor = face_tensor.to(analyzer.device)

            with torch.no_grad():
                outputs = analyzer.model(face_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(probabilities, 1)
                emotion_idx = predicted.item()
                confidence = probabilities[0][emotion_idx].item()

                emotion = EMOTION_LABELS_KK[emotion_idx]

                if emotion_screenshots[emotion] is None and confidence > 0.7:
                    emotion_screenshots[emotion] = frame.copy()

                emotion_counts[emotion] += 1

                emotion_sequence.append(emotion)
                if last_emotion is not None:
                    transition = f"{last_emotion}→{emotion}"
                    emotion_transitions[transition] = emotion_transitions.get(transition, 0) + 1
                last_emotion = emotion

                engagement_score = engagement_weights[emotion] * confidence
                engagement_scores.append(engagement_score)
                time_segments.append(current_time)

                label = f"Bet {i+1}: {emotion} ({confidence:.2f})"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        info_text = f"FPS: {fps:.1f} | Uderis: {progress_percentage:.1f}%"
        cv2.putText(frame, info_text, (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        out.write(frame)
        processed_frames += 1
        progress = processed_frames / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Өңдеу: {processed_frames}/{total_frames} кадр")

    cap.release()
    out.release()

    avg_engagement = sum(engagement_scores) / len(engagement_scores) if engagement_scores else 0
    engagement_trend = np.polyfit(time_segments, engagement_scores, 1)[0] if len(engagement_scores) >= 2 else 0.0

    emotion_changes = sum(1 for i in range(1, len(emotion_sequence)) if emotion_sequence[i] != emotion_sequence[i-1])
    emotion_stability = 1 - (emotion_changes / len(emotion_sequence)) if emotion_sequence else 0

    pdf_buffer = generate_pdf_report({
        'emotion_counts': emotion_counts,
        'total_frames': total_frames,
        'face_detection_count': face_detection_count,
        'face_coverage_distribution': face_coverage_distribution,
        'output_path': output_path,
        'duration': duration,
        'avg_engagement': avg_engagement,
        'engagement_trend': engagement_trend,
        'emotion_stability': emotion_stability,
        'emotion_transitions': emotion_transitions,
        'engagement_scores': engagement_scores,
        'time_segments': time_segments
    }, video_path, emotion_screenshots)

    return {
        'emotion_counts': emotion_counts,
        'total_frames': total_frames,
        'face_detection_count': face_detection_count,
        'face_coverage_distribution': face_coverage_distribution,
        'output_path': output_path,
        'duration': duration,
        'avg_engagement': avg_engagement,
        'engagement_trend': engagement_trend,
        'emotion_stability': emotion_stability,
        'emotion_transitions': emotion_transitions,
        'engagement_scores': engagement_scores,
        'time_segments': time_segments,
        'pdf_report': pdf_buffer
    }

if __name__ == '__main__':
    main()
