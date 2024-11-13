import pandas as pd
import csv
import sys
import os
from typing import List, Dict
import yt_dlp
import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import shutil
import statistics
import whisper
from moviepy.editor import VideoFileClip
from PIL import Image as PILImage
import torch
from transformers import CLIPProcessor, CLIPModel
import traceback
from ultralytics import YOLO  # Add this to your imports
from PyPDF2 import PdfMerger  # Add this to your imports

# Load YOLO model once at the start
yolo_model = YOLO('yolov8n.pt')  # Using the nano model for speed, can use 's', 'm', or 'l' for better accuracy

def check_ffmpeg():
    """Check if FFmpeg is installed and provide instructions if not"""
    if shutil.which('ffmpeg') is None:
        print("\nERROR: FFmpeg is not installed. Please install FFmpeg:")
        print("Windows: Download from https://www.gyan.dev/ffmpeg/builds/ and add to PATH")
        print("Mac: Run 'brew install ffmpeg'")
        print("Linux: Run 'sudo apt-get install ffmpeg' or equivalent")
        sys.exit(1)

def convert_number(value: str) -> float:
    """Convert string number (with possible comma) to float"""
    if not value:  # Handle empty values
        return 0.0
    return float(value.replace(',', ''))

def detect_scenes(video_path: str, threshold: float = 20.0, min_scene_duration: float = 1.0) -> List[tuple]:
    """
    Detect scene changes in video with scene merging
    threshold: Higher value = less sensitive
    min_scene_duration: Minimum scene duration in seconds
    """
    print("Starting scene detection...")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    print(f"Video properties: {fps} FPS, {duration:.1f}s duration")
    
    # Detect raw scenes first
    raw_scenes = []
    prev_frame = None
    current_scene_start = 0
    frame_number = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale and blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        if prev_frame is not None:
            # Calculate difference
            diff = cv2.absdiff(gray, prev_frame)
            mean_diff = np.mean(diff)
            
            if mean_diff > threshold:
                scene_time = frame_number/fps
                if raw_scenes or scene_time > 0.5:  # Skip very start of video
                    raw_scenes.append((current_scene_start, scene_time))
                current_scene_start = scene_time
        
        prev_frame = gray
        frame_number += 1
        
        # Print progress
        if frame_number % 30 == 0:
            progress = (frame_number / frame_count) * 100
            print(f"Processing: {progress:.1f}% complete", end='\r')
    
    # Add the final scene
    if current_scene_start < duration:
        raw_scenes.append((current_scene_start, duration))
    
    cap.release()
    
    # Merge short scenes
    merged_scenes = []
    current_start = None
    current_end = None
    
    for start, end in raw_scenes:
        if current_start is None:
            current_start = start
            current_end = end
        else:
            # If scene is too short, merge with previous
            if end - start < min_scene_duration:
                current_end = end
            else:
                # Add previous merged scene
                merged_scenes.append((current_start, current_end))
                current_start = start
                current_end = end
    
    # Add final merged scene
    if current_start is not None:
        merged_scenes.append((current_start, current_end))
    
    # Print scene information
    print("\nDetected scenes:")
    for i, (start, end) in enumerate(merged_scenes, 1):
        print(f"Scene {i}: {start:.1f}s - {end:.1f}s (duration: {end-start:.1f}s)")
    
    return merged_scenes

def extract_audio_script(video_path: str) -> str:
    """Extract speech from video and convert to text"""
    import time
    
    try:
        # Load the Whisper model
        model = whisper.load_model("base")
        
        # Add delay to ensure file is released
        time.sleep(2)
        
        # Extract audio with retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print("Extracting audio from video...")
                video = VideoFileClip(video_path)
                audio_path = os.path.join(os.path.dirname(video_path), "temp_audio.wav")
                video.audio.write_audiofile(audio_path, verbose=False, logger=None)
                video.close()
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Retry {attempt + 1}/{max_retries} after error: {str(e)}")
                    time.sleep(2)  # Wait before retry
                else:
                    raise
        
        # Transcribe audio
        print("Transcribing audio...")
        result = model.transcribe(audio_path)
        
        # Clean up
        try:
            os.remove(audio_path)
        except:
            pass
            
        return result["text"]
    except Exception as e:
        print(f"Error extracting script: {str(e)}")
        return "Script extraction failed"

def create_pdf_report(scenes: List[tuple], output_dir: str, reel_stats: Dict, script: str = None) -> None:
    """Create PDF report with screenshots, timing information, and script"""
    # Use the reel URL to create a more meaningful filename
    pdf_path = os.path.join(output_dir, 'scene_analysis.pdf')
    doc = SimpleDocTemplate(
        pdf_path, 
        pagesize=letter,
        leftMargin=50,      
        rightMargin=50,     
        topMargin=50,       
        bottomMargin=50     
    )
    styles = getSampleStyleSheet()
    story = []

    # Add title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    story.append(Paragraph("Reel Scene Analysis", title_style))
    
    # Add URL
    url_style = ParagraphStyle(
        'URL',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    story.append(Paragraph(f"URL: {reel_stats.get('url', '')}", url_style))
    story.append(PageBreak())

    # Add statistics
    stats_style = ParagraphStyle(
        'Stats',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=20,
        alignment=1  # Center alignment
    )
    stats_text = f"""
    <b>Reel Statistics:</b><br/>
    Views: {reel_stats.get('views', 0):,.0f}<br/>
    Likes: {reel_stats.get('likes', 0):,.0f}<br/>
    Comments: {reel_stats.get('comments', 0):,.0f}<br/>
    URL: {reel_stats.get('url', '')}<br/>
    """
    story.append(Paragraph(stats_text, stats_style))
    story.append(PageBreak())

    # Create scene entries
    for i, (start, end) in enumerate(scenes, 1):
        try:
            duration = end - start
            screenshot_pattern = f"scene_{i:02d}_{start:.1f}s-{end:.1f}s.jpg"
            img_path = os.path.join(output_dir, screenshot_pattern)
            
            if os.path.exists(img_path):
                frame = cv2.imread(img_path)
                if frame is not None:
                    content_info = analyze_image_content(frame)
                    style_info = analyze_video_style(frame)
                    
                    # Create detailed scene description with more padding
                    scene_text = f"""
                    <para alignment="left" spaceAfter="10">
                    <b>Scene {i}</b><br/><br/>
                    Time: {start:.1f}s - {end:.1f}s<br/>
                    Duration: {duration:.1f}s<br/><br/>
                    <b>Visual Analysis:</b><br/>
                    Composition: {content_info['composition']}<br/>
                    Main Object: {content_info['main_object']}<br/>
                    Detected Objects: {content_info['all_objects']}<br/>
                    Confidence: {content_info['confidence']:.2f}<br/><br/>
                    <b>Style Analysis:</b><br/>
                    Color Style: {style_info['color_style']}<br/>
                    Lighting: {style_info['lighting']}<br/>
                    Contrast: {style_info['contrast_level']}<br/>
                    Composition: {style_info['composition']}<br/>
                    </para>
                    """
                    
                    # Create table with text and image
                    img = ReportLabImage(img_path)
                    img._restrictSize(350, 250)  # Slightly smaller images
                    
                    # Create table with better spacing
                    data = [[Paragraph(scene_text, styles['Normal']), img]]
                    table = Table(data, colWidths=[300, 350])  # Adjusted column widths
                    table.setStyle(TableStyle([
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('TOPPADDING', (0, 0), (-1, -1), 20),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 20),
                        ('LEFTPADDING', (0, 0), (-1, -1), 20),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 20),
                    ]))
                    
                    story.append(table)
                    story.append(PageBreak())
                    
                    print(f"Added scene {i} to PDF")
        except Exception as e:
            print(f"Error processing scene {i} for PDF: {str(e)}")
            traceback.print_exc()
            continue

    # Add script section if available
    if script and script.strip():
        script_style = ParagraphStyle(
            'ScriptStyle',
            parent=styles['Normal'],
            fontSize=11,
            leading=14,
            leftIndent=20,
            rightIndent=20,
            spaceAfter=20
        )
        story.append(Paragraph("<b>Video Script:</b>", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        # Split script into paragraphs for better formatting
        script_paragraphs = script.split('\n')
        for paragraph in script_paragraphs:
            if paragraph.strip():  # Only add non-empty paragraphs
                story.append(Paragraph(paragraph, script_style))
                story.append(Spacer(1, 8))
    
    # Generate PDF
    try:
        if len(story) > 1:
            doc.build(story)
            print(f"PDF report generated successfully: {pdf_path}")
        else:
            print("Error: No content to generate PDF")
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        traceback.print_exc()

def analyze_image_content(frame) -> Dict[str, str]:
    """
    Analyze image content using YOLO object detection
    Returns detailed description and detected objects
    """
    try:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run YOLO detection
        results = yolo_model(frame_rgb)
        
        # Get detected objects with confidence scores
        detected_objects = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = results[0].names[class_id]
                if confidence > 0.3:  # Only include objects with >30% confidence
                    detected_objects.append({
                        'name': class_name,
                        'confidence': confidence
                    })
        
        # Sort by confidence
        detected_objects.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Analyze composition based on object placement
        if len(detected_objects) > 0:
            main_object = detected_objects[0]['name']
            confidence = detected_objects[0]['confidence']
            
            # Create object list string
            object_list = ', '.join([f"{obj['name']} ({obj['confidence']:.2f})" 
                                   for obj in detected_objects[:3]])  # Top 3 objects
            
            # Determine shot type based on objects
            if len(detected_objects) > 2:
                composition = "complex scene with multiple objects"
            else:
                composition = f"focused shot of {main_object}"
                
        else:
            main_object = "no clear objects detected"
            confidence = 0.0
            object_list = "none detected"
            composition = "minimal composition"
            
        return {
            'composition': composition,
            'main_object': main_object,
            'confidence': confidence,
            'all_objects': object_list
        }
        
    except Exception as e:
        print(f"Error in content analysis: {str(e)}")
        return {
            'composition': 'unknown',
            'main_object': 'unknown',
            'confidence': 0.0,
            'all_objects': 'analysis failed'
        }

def save_scene_screenshots(video_path: str, scenes: List[tuple], output_dir: str, reel_stats: Dict) -> None:
    """Save middle frame of each scene as screenshot"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print("\nSaving screenshots...")
    for i, (start, end) in enumerate(scenes, 1):
        # Take screenshot from middle of scene
        middle_time = (start + end) / 2
        middle_frame = int(middle_time * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame = cap.read()
        
        if ret:
            filename = f"scene_{i:02d}_{start:.1f}s-{end:.1f}s.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            print(f"Saved screenshot {i}: {filename}")
    
    cap.release()
    
    # Now create the PDF report
    create_pdf_report(scenes, output_dir, reel_stats)

def download_and_analyze_reel(url: str, output_dir: str, reel_stats: Dict) -> None:
    """Download reel and analyze its scenes"""
    try:
        # Extract reel ID from URL and get title
        reel_id = url.split('/')[-2]  # Get the ID from the URL
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            reel_title = f"{info['title']}_{reel_id}"  # Add reel ID to title
        
        # Create unique folder for this reel
        reel_dir = os.path.join(output_dir, reel_title)
        if not os.path.exists(reel_dir):
            os.makedirs(reel_dir)
        
        # Download the reel with unique filename
        ydl_opts = {
            'format': 'best',
            'outtmpl': os.path.join(reel_dir, f'%(title)s_{reel_id}.%(ext)s'),
            'quiet': False,
            'no_warnings': True,
        }
        
        video_path = None
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_path = ydl.prepare_filename(info)
            print(f"\nSuccessfully downloaded reel to: {reel_dir}")
            
            # Make sure the video file exists before processing
            if os.path.exists(video_path):
                print("\nAnalyzing scenes...")
                scenes = detect_scenes(video_path)
                
                # Extract script
                print("\nExtracting video script...")
                script = extract_audio_script(video_path)
                
                # Save screenshots and create PDF
                save_scene_screenshots(video_path, scenes, reel_dir, reel_stats)
                create_pdf_report(scenes, reel_dir, reel_stats, script)  # Pass the script here
                
                print(f"\nScene analysis complete! Found {len(scenes)} scenes")
                print(f"Results saved in: {reel_dir}")
            else:
                print(f"Error: Downloaded video not found at {video_path}")
        
        finally:
            # Clean up video file
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
                print(f"Cleaned up video file: {video_path}")
        
    except Exception as e:
        print(f"Error processing reel: {str(e)}")

def merge_pdf_reports(output_dir: str) -> None:
    """Merge all PDF reports in the directory into one combined report"""
    try:
        merger = PdfMerger()
        pdf_files = []
        
        # Find all PDF files in subdirectories
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith('scene_analysis.pdf'):
                    pdf_path = os.path.join(root, file)
                    pdf_files.append(pdf_path)
        
        if not pdf_files:
            print("No PDF files found to merge")
            return
            
        # Sort PDFs by creation time
        pdf_files.sort(key=lambda x: os.path.getctime(x))
        
        # Merge PDFs
        for pdf in pdf_files:
            merger.append(pdf)
            
        # Use the directory name (from CSV) for the final PDF
        csv_name = os.path.basename(output_dir)
        merged_path = os.path.join(output_dir, f'{csv_name}.pdf')
        merger.write(merged_path)
        merger.close()
        
        print(f"\nMerged {len(pdf_files)} PDF reports into: {merged_path}")
        
    except Exception as e:
        print(f"Error merging PDFs: {str(e)}")
        traceback.print_exc()

def process_reels(csv_path: str, min_views: int = 0, num_reels: int = None) -> None:
    """
    Process multiple reels and create combined report
    
    Args:
        csv_path: Path to CSV file containing reel URLs
        min_views: Minimum number of views to process (default: 0)
        num_reels: Number of reels to process (default: None = all reels)
    """
    # Create output directory based on CSV name
    output_dir = os.path.splitext(csv_path)[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Read URLs from CSV
    try:
        # Read CSV with proper column names
        df = pd.read_csv(csv_path)
        total_reels = len(df)
        print(f"\nFound {total_reels} total reels in CSV")
        
        # Filter by minimum views
        if min_views > 0:
            df = df[df['Views'] >= min_views]
            filtered_reels = len(df)
            print(f"Found {filtered_reels} reels with {min_views:,}+ views")
        
        # Sort by views descending
        df = df.sort_values('Views', ascending=False)
        
        # Limit number of reels if specified
        if num_reels and num_reels < len(df):
            df = df.head(num_reels)
            print(f"Will process top {num_reels} reels")
        else:
            print(f"Will process all {len(df)} qualifying reels")
        
        # Get final list of URLs
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
        
        # After processing all reels, merge the PDFs
        merge_pdf_reports(output_dir)
            
    except Exception as e:
        print(f"Error reading CSV: {str(e)}")
        traceback.print_exc()
        return

def analyze_video_style(frame) -> Dict[str, str]:
    """Analyze video style (color, lighting, composition)"""
    try:
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Analyze brightness (V channel)
        v_channel = hsv[:,:,2]
        mean_brightness = np.mean(v_channel)
        std_brightness = np.std(v_channel)
        
        # Determine lighting
        if mean_brightness > 200:
            lighting = "High-key/Bright"
        elif mean_brightness > 150:
            lighting = "Well-lit"
        elif mean_brightness > 100:
            lighting = "Medium"
        else:
            lighting = "Low-key/Dark"
            
        # Analyze contrast
        if std_brightness > 60:
            contrast_level = "High contrast"
        elif std_brightness > 30:
            contrast_level = "Medium contrast"
        else:
            contrast_level = "Low contrast"
            
        # Analyze color distribution
        s_channel = hsv[:,:,1]
        mean_saturation = np.mean(s_channel)
        
        if mean_saturation > 150:
            color_style = "Vibrant/Saturated"
        elif mean_saturation > 100:
            color_style = "Natural colors"
        elif mean_saturation > 50:
            color_style = "Muted colors"
        else:
            color_style = "Minimal/Clean"
            
        # Basic composition analysis
        height, width = frame.shape[:2]
        center_region = frame[height//4:3*height//4, width//4:3*width//4]
        center_brightness = np.mean(center_region)
        edge_brightness = np.mean(frame) - center_brightness
        
        if abs(edge_brightness) > 30:
            composition = "Vignette/Focused"
        elif std_brightness < 20:
            composition = "Studio/Clean"
        else:
            composition = "Natural/Documentary"
        
        return {
            'lighting': lighting,
            'contrast_level': contrast_level,
            'color_style': color_style,
            'composition': composition
        }
        
    except Exception as e:
        print(f"Error in style analysis: {str(e)}")
        return {
            'lighting': 'unknown',
            'contrast_level': 'unknown',
            'color_style': 'unknown',
            'composition': 'unknown'
        }

def main():
    # Check for FFmpeg first
    check_ffmpeg()
    
    # Configuration
    CSV_PATH = 'elbowdesign reels.csv'
    MIN_VIEWS = 1000000  # minimum 100k views
    NUM_REELS = 25  # Set to None for all reels, or a number to limit
    
    process_reels(CSV_PATH, MIN_VIEWS, NUM_REELS)

if __name__ == "__main__":
    main()
