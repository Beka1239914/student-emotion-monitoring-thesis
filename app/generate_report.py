import os
from datetime import datetime
from fpdf import FPDF
import plotly.express as px
import pandas as pd

EMOTION_LABELS_KK = {
    'Angry': 'Ашулы',
    'Disgust': 'Жиіркену',
    'Fear': 'Қорқыныш',
    'Happy': 'Қуаныш',
    'Sad': 'Қайғы',
    'Surprise': 'Таңданыс',
    'Neutral': 'Бейтарап',
}

FONT_REGULAR = "C:\\Windows\\Fonts\\arial.ttf"
FONT_BOLD = "C:\\Windows\\Fonts\\arialbd.ttf"


def generate_pdf_report(df: pd.DataFrame, summary: dict, video_path: str) -> str:
    """
    Generate a PDF report with emotion analysis results in Kazakh.

    Args:
        df: DataFrame containing emotion analysis results
        summary: Dictionary containing emotion statistics
        video_path: Path to the analyzed video file

    Returns:
        Path to the generated PDF report
    """
    pdf = FPDF()
    pdf.add_page()

    pdf.add_font("ArialKK", "", FONT_REGULAR)
    pdf.add_font("ArialKK", "B", FONT_BOLD)

    # Title
    pdf.set_font("ArialKK", "B", 16)
    pdf.cell(0, 10, "Сату қоңырауларын эмоциялық талдау есебі", ln=True, align="C")
    pdf.ln(10)

    # Timestamp and file info
    pdf.set_font("ArialKK", "", 12)
    pdf.cell(0, 10, f"Жасалған күні: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 10, f"Бейне файл: {os.path.basename(video_path)}", ln=True)
    pdf.ln(10)

    # Summary statistics
    pdf.set_font("ArialKK", "B", 14)
    pdf.cell(0, 10, "Жалпы статистика", ln=True)
    pdf.set_font("ArialKK", "", 12)
    pdf.cell(0, 10, f"Талданған кадрлар саны: {summary['total_frames']}", ln=True)
    pdf.ln(5)

    # Emotion distribution
    pdf.set_font("ArialKK", "B", 14)
    pdf.cell(0, 10, "Эмоциялар үлестірімі", ln=True)
    pdf.set_font("ArialKK", "", 12)

    for emotion, percentage in summary['emotion_percentages'].items():
        label = EMOTION_LABELS_KK.get(emotion, emotion)
        pdf.cell(0, 10, f"{label}: {percentage:.1f}%", ln=True)

    pdf.ln(10)

    # Emotion timeline chart
    fig = px.line(df, x='timestamp', y='emotion',
                  title='Уақыт бойынша эмоция өзгерістері',
                  labels={'timestamp': 'Уақыт (секунд)', 'emotion': 'Эмоция'})

    img_bytes = fig.to_image(format="png")
    img_path = "reports/timeline.png"
    with open(img_path, "wb") as f:
        f.write(img_bytes)

    pdf.image(img_path, x=10, y=pdf.get_y(), w=190)
    pdf.ln(150)

    # Emotion distribution bar chart
    emotion_df = pd.DataFrame({
        'Эмоция': [EMOTION_LABELS_KK.get(e, e) for e in summary['emotion_percentages'].keys()],
        'Пайыз': list(summary['emotion_percentages'].values())
    })
    fig = px.bar(emotion_df, x='Эмоция', y='Пайыз',
                 title='Эмоциялар үлестірімі',
                 labels={'Пайыз': 'Пайыз (%)'})

    img_bytes = fig.to_image(format="png")
    img_path = "reports/distribution.png"
    with open(img_path, "wb") as f:
        f.write(img_bytes)

    pdf.image(img_path, x=10, y=pdf.get_y(), w=190)

    # Save PDF
    report_path = "reports/emotion_analysis_report.pdf"
    pdf.output(report_path)

    os.remove("reports/timeline.png")
    os.remove("reports/distribution.png")

    return report_path
