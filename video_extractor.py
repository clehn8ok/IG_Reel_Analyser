import pandas as pd
import os
from typing import List, Dict
import yt_dlp
import cv2
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import shutil
import whisper
from moviepy.editor import VideoFileClip
import traceback
from PyPDF2 import PdfMerger
import easyocr  # pip install easyocr

# Initialize EasyOCR reader (this will download the model on first run)
reader = easyocr.Reader(['en'])

def check_ffmpeg():
    """Check if FFmpeg is installed and provide instructions if not"""
    if shutil.which('ffmpeg') is None:
        print("\nERROR: FFmpeg is not installed. Please install FFmpeg:")
        print("Windows: Download from https://www.gyan.dev/ffmpeg/builds/ and add to PATH")
        print("Mac: Run 'brew install ffmpeg'")
        print("Linux: Run 'sudo apt-get install ffmpeg' or equivalent")
        sys.exit(1)

def extract_audio_script(video_path: str) -> str:
    """Extract and transcribe audio from video"""
    try:
        print("\nExtracting audio from video...")
        video = VideoFileClip(video_path)
        
        print("Transcribing audio...")
        model = whisper.load_model("base")
        result = model.transcribe(video_path)
        
        video.close()
        return result["text"].strip()
        
    except Exception as e:
        print(f"Error extracting audio: {str(e)}")
        return None

def create_pdf_report(output_dir: str, reel_stats: Dict, script: str = None) -> None:
    """Create PDF report with reel title and transcription"""
    pdf_path = os.path.join(output_dir, 'text_analysis.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, leftMargin=50, rightMargin=50, topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24, spaceAfter=30, alignment=1)
    info_style = ParagraphStyle('Info', parent=styles['Normal'], fontSize=10, spaceAfter=10)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=14, spaceAfter=10)
    
    story = []
    
    # Title and metadata
    story.append(Paragraph("Reel Analysis", title_style))
    #story.append(Paragraph(f"<b>Title:</b> {reel_stats.get('title', 'Untitled')}", info_style))
    story.append(Paragraph(f"<b>Description:</b> {reel_stats.get('description', '')}", info_style))
    #story.append(Paragraph(f"<b>URL:</b> {reel_stats.get('url', '')}", info_style))
    story.append(Paragraph(f"<b>Views:</b> {reel_stats.get('views', 0):,}", info_style))
    
    # Add transcription section
    if script:
        story.append(Spacer(1, 20))
        story.append(Paragraph("Voice Over", heading_style))
        story.append(Paragraph(script, info_style))
    
    # Generate PDF
    try:
        doc.build(story)
        print(f"PDF report generated successfully: {pdf_path}")
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        traceback.print_exc()

def download_and_analyze_reel(url: str, output_dir: str, reel_stats: Dict) -> None:
    """Download and analyze a single reel"""
    video_path = None
    
    try:
        # Extract reel ID from URL
        reel_id = url.split('/')[-2]  # Gets the ID from the URL
        reel_dir = os.path.join(output_dir, f'reel_{reel_id}')
        
        if not os.path.exists(reel_dir):
            os.makedirs(reel_dir)
        
        # Configure yt-dlp options with the specific directory
        ydl_opts = {
            'format': 'best',
            'outtmpl': os.path.join(reel_dir, 'video.%(ext)s'),  # Simplified output template
            'quiet': True,
            'extract_flat': False,
            'force_generic_extractor': False,
            'add_header': [
                'User-Agent:Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            ],
            # Remove cookie handling for now
            'no_warnings': True,
            'ignoreerrors': True
        }
        
        # Download video and get metadata
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                # First get info
                info = ydl.extract_info(url, download=True)  # Changed to True to combine info and download
                
                if info:
                    # Update reel stats with metadata
                    reel_stats.update({
                        'title': info.get('title', 'Untitled'),
                        'description': info.get('description', ''),
                        'uploader': info.get('uploader', ''),
                        'upload_date': info.get('upload_date', ''),
                    })
                    
                    print(f"\nReel Title: {reel_stats['title']}")
                    print(f"Reel Description: {reel_stats['description']}")
                    
                    # Get the actual video path
                    video_path = os.path.join(reel_dir, 'video.' + info.get('ext', 'mp4'))
                    
                    if os.path.exists(video_path):
                        print(f"\nSuccessfully downloaded reel to: {reel_dir}")
                        
                        # Extract audio script
                        script = extract_audio_script(video_path)
                        
                        # Create PDF with reel info and script
                        create_pdf_report(reel_dir, reel_stats, script)
                        
                        print(f"Results saved in: {reel_dir}")
                    else:
                        print(f"Error: Downloaded video not found at {video_path}")
                else:
                    print(f"Error: Could not extract info from {url}")
                
            except Exception as e:
                print(f"Error extracting metadata: {str(e)}")
                traceback.print_exc()
        
    except Exception as e:
        print(f"Error processing reel {url}: {str(e)}")
        traceback.print_exc()
        
    finally:
        # Clean up video file if it exists
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
                print(f"Cleaned up video file: {video_path}")
            except Exception as e:
                print(f"Error cleaning up video file: {str(e)}")

def process_reels(csv_path: str, min_views: int = 0, num_reels: int = None) -> None:
    """Process multiple reels and create combined report"""
    output_dir = os.path.splitext(csv_path)[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    try:
        df = pd.read_csv(csv_path)
        total_reels = len(df)
        print(f"\nFound {total_reels} total reels in CSV")
        
        if min_views > 0:
            df = df[df['Views'] >= min_views]
            print(f"Found {len(df)} reels with {min_views:,}+ views")
        
        df = df.sort_values('Views', ascending=False)
        
        if num_reels and num_reels < len(df):
            df = df.head(num_reels)
            print(f"Will process top {num_reels} reels")
        else:
            print(f"Will process all {len(df)} qualifying reels")
        
        urls = df['Reel URL'].tolist()
        
        if not urls:
            print("No reels found matching criteria")
            return

        print("\nView counts of reels to be processed:")
        for i, (_, row) in enumerate(df.iterrows(), 1):
            print(f"Reel {i}: {row['Views']:,} views")
        
        print(f"\nStarting processing of {len(urls)} reels...")
        
        for i, url in enumerate(urls, 1):
            try:
                views = df[df['Reel URL'] == url]['Views'].iloc[0]
                print(f"\nProcessing viral reel {i} of {len(urls)} ({views:,.0f} views)...")
                
                reel_stats = {
                    'url': url,
                    'views': views
                }
                
                download_and_analyze_reel(url, output_dir, reel_stats)
                
            except Exception as e:
                print(f"Error processing reel {url}: {str(e)}")
                traceback.print_exc()
                continue
        
        merge_pdf_reports(output_dir)
            
    except Exception as e:
        print(f"Error reading CSV: {str(e)}")
        traceback.print_exc()

def merge_pdf_reports(output_dir: str) -> None:
    """Merge all PDF reports into a single file"""
    try:
        merger = PdfMerger()
        pdf_files = []
        
        # Find all PDF files in subdirectories
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith('text_analysis.pdf'):
                    pdf_path = os.path.join(root, file)
                    pdf_files.append(pdf_path)
        
        if not pdf_files:
            print("No PDF files found to merge")
            return
            
        # Sort PDF files by directory name to maintain order
        pdf_files.sort()
        
        # Add each PDF to the merger
        for pdf in pdf_files:
            try:
                merger.append(pdf)
            except Exception as e:
                print(f"Error adding PDF {pdf}: {str(e)}")
                continue
        
        # Write the merged PDF one folder above
        parent_dir = os.path.dirname(output_dir)
        output_path = os.path.join(parent_dir, f'{os.path.basename(output_dir)}.pdf')
        merger.write(output_path)
        merger.close()
        
        print(f"\nMerged {len(pdf_files)} PDF reports into: {output_path}")
        
    except Exception as e:
        print(f"Error merging PDFs: {str(e)}")
        traceback.print_exc()

def main():
    check_ffmpeg()
    
    # Get all CSV files in the current directory
    csv_files = [f for f in os.listdir() if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in current directory")
        return
        
    print(f"Found {len(csv_files)} CSV files to process:")
    for csv_file in csv_files:
        print(f"- {csv_file}")
    
    MIN_VIEWS = 100000  # minimum 100k views
    NUM_REELS = 25  # Set to None for all reels, or a number to limit
    
    # Process each CSV file
    for csv_file in csv_files:
        print(f"\n{'='*50}")
        print(f"Processing {csv_file}...")
        print(f"{'='*50}")
        
        try:
            process_reels(csv_file, MIN_VIEWS, NUM_REELS)
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
            traceback.print_exc()
            continue
        
    print("\nAll CSV files processed!")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
