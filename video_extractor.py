import csv
import sys
import os
from typing import List, Dict
import yt_dlp
import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import shutil
import statistics

def convert_number(value: str) -> float:
    """Convert string number (with possible comma) to float"""
    if not value:  # Handle empty values
        return 0.0
    return float(value.replace(',', ''))

def detect_scenes(video_path: str, threshold: float = 30.0, min_scene_duration: float = 0.5) -> List[tuple]:
    """
    Detect scene changes in video
    threshold: Higher value = less sensitive (range typically 20-60)
    min_scene_duration: Minimum scene duration in seconds
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    scenes = []
    prev_frame = None
    current_scene_start = 0
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale for comparison
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is not None:
            # Calculate difference between frames
            diff = cv2.absdiff(gray, prev_frame)
            mean_diff = np.mean(diff)
            
            # If difference is above threshold, we found a scene change
            if mean_diff > threshold and (frame_number/fps - current_scene_start) >= min_scene_duration:
                scenes.append((current_scene_start, frame_number/fps))
                current_scene_start = frame_number/fps
        
        prev_frame = gray
        frame_number += 1
    
    # Add the final scene
    scenes.append((current_scene_start, duration))
    
    cap.release()
    return scenes

def create_pdf_report(scenes: List[tuple], output_dir: str, reel_stats: Dict) -> None:
    """Create PDF report with screenshots and timing information"""
    pdf_path = os.path.join(output_dir, 'scene_analysis.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Get page dimensions with margins
    page_width, page_height = letter
    margin = 50  # points
    usable_width = page_width - (2 * margin)
    usable_height = page_height - (2 * margin)

    # Add title page content
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    story.append(Paragraph("Reel Scene Analysis", title_style))
    
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
    Views: {reel_stats['views']:,.0f}<br/>
    Likes: {reel_stats['likes']:,.0f}<br/>
    Comments: {reel_stats['comments']:,.0f}<br/>
    URL: {reel_stats['url']}<br/>
    """
    story.append(Paragraph(stats_text, stats_style))
    story.append(PageBreak())

    # Create scene entries
    for i, (start, end) in enumerate(scenes, 1):
        duration = end - start
        scene_text = f"""
        <b>Scene {i}</b><br/>
        Time: {start:.1f}s - {end:.1f}s<br/>
        Duration: {duration:.1f}s<br/>
        """
        
        img_path = os.path.join(output_dir, f'scene_{start:.1f}s-{end:.1f}s.jpg')
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            height, width = img.shape[:2]
            aspect = width / height
            
            # Make image smaller (60% of page height)
            target_height = usable_height * 0.6
            if aspect > 1:  # Landscape
                img_width = min(usable_width, target_height * aspect)
                img_height = img_width / aspect
            else:  # Portrait
                img_height = min(target_height, usable_width / aspect)
                img_width = img_height * aspect

            # Create table with text and image
            img_element = Image(img_path, width=img_width, height=img_height)
            data = [[Paragraph(scene_text, stats_style)], [img_element]]
            
            # Create table with specific dimensions
            table = Table(data, colWidths=[usable_width])
            table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('TOPPADDING', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ]))
            
            story.append(table)
            
            if i < len(scenes):
                story.append(PageBreak())

    # Generate PDF
    try:
        doc.build(story)
        print(f"PDF report generated: {pdf_path}")
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        try:
            # Reduce image sizes by additional 20% if needed
            for item in story:
                if isinstance(item, Table):
                    img = item._cellvalues[1][0]
                    if isinstance(img, Image):
                        img.drawWidth *= 0.8
                        img.drawHeight *= 0.8
            doc.build(story)
            print(f"PDF generated with reduced image sizes: {pdf_path}")
        except Exception as e:
            print(f"Failed to generate PDF even with reduced sizes: {str(e)}")

def save_scene_screenshots(video_path: str, scenes: List[tuple], output_dir: str, reel_stats: Dict) -> None:
    """Save middle frame of each scene as screenshot"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    
    # Save timing information to text file
    with open(os.path.join(output_dir, 'scene_times.txt'), 'w') as f:
        # Write reel statistics
        f.write(f"Reel Statistics:\n")
        f.write(f"Views: {reel_stats['views']:,.0f}\n")
        f.write(f"Likes: {reel_stats['likes']:,.0f}\n")
        f.write(f"Comments: {reel_stats['comments']:,.0f}\n")
        f.write(f"URL: {reel_stats['url']}\n\n")
        
        # Write video duration and scene information
        f.write(f"Total video duration: {total_duration:.1f}s\n\n")
        for i, (start, end) in enumerate(scenes, 1):
            duration = end - start
            f.write(f"Scene {i}:\n")
            f.write(f"  Time: {start:.1f}s-{end:.1f}s\n")
            f.write(f"  Duration: {duration:.1f}s\n\n")
            
            # Calculate middle frame of scene
            middle_time = (start + end) / 2
            middle_frame = int(middle_time * fps)
            
            # Set frame position and read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
            ret, frame = cap.read()
            
            if ret:
                filename = f"scene_{start:.1f}s-{end:.1f}s.jpg"
                cv2.imwrite(os.path.join(output_dir, filename), frame)
    
    cap.release()
    
    # Create PDF report
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
                scenes = detect_scenes(video_path, threshold=20.0, min_scene_duration=0.5)
                
                # Save screenshots and timing info in the same directory
                save_scene_screenshots(video_path, scenes, reel_dir, reel_stats)
                
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

def analyze_viral_reels(csv_path: str, min_views: float = 100000, num_reels: int = None) -> None:
    """Analyze reels from CSV file and process viral reels"""
    reels: List[Dict] = []
    processed_ids = set()
    
    output_dir = os.path.splitext(os.path.basename(csv_path))[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create CSV with statistics
    stats_csv = os.path.join(output_dir, 'reel_statistics.csv')
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames + [
                'Average Clip Duration', 
                'Median Clip Duration',
                'Likes per View',
                'Comments per View'
            ]
            
            # Read all rows and store them
            rows = []
            for row in reader:
                views = convert_number(row['Views'])
                likes = convert_number(row['Likes'])
                comments = convert_number(row['Comments'])
                
                # Calculate engagement metrics
                row['Likes per View'] = f"{(likes/views)*100:.2f}%" if views > 0 else "N/A"
                row['Comments per View'] = f"{(comments/views)*100:.2f}%" if views > 0 else "N/A"
                
                if views >= min_views:
                    reel = {
                        'url': row['Reel URL'],
                        'views': views,
                        'likes': likes,
                        'comments': comments
                    }
                    reels.append(reel)
                rows.append(row)
        
        if not reels:
            print(f"\nNo reels found with {min_views:,.0f}+ views")
            return
            
        sorted_reels = sorted(reels, key=lambda x: x['views'], reverse=True)
        
        # Limit number of reels if specified
        if num_reels is not None:
            sorted_reels = sorted_reels[:num_reels]
            print(f"\nProcessing top {num_reels} viral reels")
        else:
            print(f"\nProcessing all {len(sorted_reels)} viral reels")
        
        # Process selected reels and track their IDs
        for i, reel in enumerate(sorted_reels, 1):
            print(f"\nProcessing viral reel {i} of {len(sorted_reels)} ({reel['views']:,.0f} views)...")
            download_and_analyze_reel(reel['url'], output_dir, reel)
            reel_id = reel['url'].split('/')[-2]
            processed_ids.add(reel_id)
        
        # Write statistics CSV after processing
        with open(stats_csv, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in rows:
                if 'Reel URL' in row and row['Reel URL']:
                    reel_id = row['Reel URL'].split('/')[-2]
                    
                    if reel_id in processed_ids:
                        matching_dirs = [d for d in os.listdir(output_dir) if reel_id in d]
                        if matching_dirs:
                            reel_dir = os.path.join(output_dir, matching_dirs[0])
                            scene_file = os.path.join(reel_dir, 'scene_times.txt')
                            
                            if os.path.exists(scene_file):
                                durations = []
                                with open(scene_file, 'r') as sf:
                                    for line in sf:
                                        if 'Duration:' in line:
                                            try:
                                                duration = float(line.split(':')[1].strip().replace('s', ''))
                                                durations.append(duration)
                                            except (ValueError, IndexError):
                                                continue
                                
                                if durations:
                                    # Add 's' postfix for seconds
                                    row['Average Clip Duration'] = f"{sum(durations) / len(durations):.1f}s"
                                    row['Median Clip Duration'] = f"{statistics.median(durations):.1f}s"
                                else:
                                    row['Average Clip Duration'] = "N/A"
                                    row['Median Clip Duration'] = "N/A"
                        else:
                            row['Average Clip Duration'] = "N/A"
                            row['Median Clip Duration'] = "N/A"
                    else:
                        row['Average Clip Duration'] = "N/A"
                        row['Median Clip Duration'] = "N/A"
                
                writer.writerow(row)
        
        print(f"\nStatistics CSV generated: {stats_csv}")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")

def main():
    # Configuration
    CSV_PATH = 'kismet_design reels.csv'  # Replace with your CSV path
    MIN_VIEWS = 100000 # minimum 100k views
    NUM_REELS = 25  # Set to None for all reels, or a number to limit
    
    analyze_viral_reels(CSV_PATH, MIN_VIEWS, NUM_REELS)

if __name__ == "__main__":
    main()
