"""
PDF Text and Image Extractor
Extract all text and images from PDF files
"""

import os
from pathlib import Path
from pypdf import PdfReader
import pdfplumber
from PIL import Image
import io


def extract_pdf_content(pdf_path: str, output_dir: str = "extracted_content"):
    """
    Extract all text and images from a PDF file
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted images
    
    Returns:
        dict with text and images info
    """
    pdf_name = Path(pdf_path).stem
    images_dir = os.path.join(output_dir, pdf_name, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    print(f"  Processing: {pdf_name}")
    
    # Extract text using pdfplumber
    all_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, 1):
                # Get page dimensions
                page_height = page.height
                
                # Define content area (exclude header/footer regions)
                # Skip top 10% and bottom 10% of page (typical header/footer area)
                top_margin = page_height * 0.1
                bottom_margin = page_height * 0.9
                
                # Crop page to main content area
                content_bbox = (0, top_margin, page.width, bottom_margin)
                cropped_page = page.within_bbox(content_bbox)
                
                # Extract text from content area
                text = cropped_page.extract_text()
                
                if text:
                    # Clean up text
                    lines = text.split('\n')
                    cleaned_lines = []
                    
                    for line in lines:
                        line = line.strip()
                        # Skip empty lines
                        if not line:
                            continue
                        # Skip lines that are only page numbers
                        if line.isdigit() and len(line) <= 3:
                            continue
                        # Skip very short lines (likely artifacts)
                        if len(line) < 3:
                            continue
                        
                        cleaned_lines.append(line)
                    
                    if cleaned_lines:
                        # Join lines with proper spacing
                        page_text = '\n'.join(cleaned_lines)
                        all_text.append(page_text)
                        
    except Exception as e:
        print(f"  Error extracting text: {e}")
        return None
    
    full_text = "\n\n".join(all_text)
    
    # Save text
    text_file = os.path.join(output_dir, pdf_name, f"{pdf_name}_text.txt")
    os.makedirs(os.path.dirname(text_file), exist_ok=True)
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(full_text)
    
    # Extract images using pypdf
    image_paths = []
    image_count = 0
    
    try:
        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages, 1):
            try:
                if '/XObject' not in page['/Resources']:
                    continue
                    
                xObject = page['/Resources']['/XObject'].get_object()
                
                for obj_name in xObject:
                    obj = xObject[obj_name]
                    
                    if obj['/Subtype'] != '/Image':
                        continue
                    
                    try:
                        # Get image size
                        width = obj['/Width']
                        height = obj['/Height']
                        
                        # Get raw image data
                        data = obj.get_data()
                        
                        # Determine color mode
                        if '/ColorSpace' in obj:
                            color_space = obj['/ColorSpace']
                            if isinstance(color_space, list):
                                color_space = color_space[0]
                            
                            if color_space == '/DeviceRGB':
                                mode = 'RGB'
                            elif color_space == '/DeviceGray':
                                mode = 'L'
                            elif color_space == '/DeviceCMYK':
                                mode = 'CMYK'
                            else:
                                mode = 'RGB'
                        else:
                            mode = 'RGB'
                        
                        # Save image
                        image_count += 1
                        
                        # Check if it's already JPEG
                        if '/Filter' in obj and obj['/Filter'] == '/DCTDecode':
                            # Save directly as JPEG
                            image_filename = f"page_{page_num}_img_{image_count}.jpg"
                            image_path = os.path.join(images_dir, image_filename)
                            with open(image_path, 'wb') as img_file:
                                img_file.write(data)
                        else:
                            # Convert to PNG using PIL
                            try:
                                # Try to create image from raw data
                                if mode == 'L':
                                    img = Image.frombytes('L', (width, height), data)
                                elif mode == 'RGB':
                                    img = Image.frombytes('RGB', (width, height), data)
                                elif mode == 'CMYK':
                                    img = Image.frombytes('CMYK', (width, height), data)
                                    img = img.convert('RGB')
                                else:
                                    img = Image.frombytes('RGB', (width, height), data)
                                
                                image_filename = f"page_{page_num}_img_{image_count}.png"
                                image_path = os.path.join(images_dir, image_filename)
                                img.save(image_path, 'PNG')
                            except Exception as e2:
                                # If frombytes fails, try opening as is
                                try:
                                    img = Image.open(io.BytesIO(data))
                                    image_filename = f"page_{page_num}_img_{image_count}.png"
                                    image_path = os.path.join(images_dir, image_filename)
                                    img.save(image_path, 'PNG')
                                except:
                                    # Last resort: save raw data
                                    image_filename = f"page_{page_num}_img_{image_count}.bin"
                                    image_path = os.path.join(images_dir, image_filename)
                                    with open(image_path, 'wb') as img_file:
                                        img_file.write(data)
                                    continue
                        
                        image_paths.append(image_path)
                        
                    except Exception as e:
                        print(f"    Warning: Could not extract image from page {page_num}: {e}")
            except Exception as e:
                # Page has no resources
                pass
                
    except Exception as e:
        print(f"  Error extracting images: {e}")
    
    return {
        "pdf_name": pdf_name,
        "text_file": text_file,
        "text_content": full_text,
        "total_pages": total_pages,
        "images": image_paths,
        "image_count": image_count
    }


def extract_all_pdfs_in_folder(folder_path: str, output_dir: str = "extracted_content"):
    """
    Extract text and images from all PDF files in a folder
    
    Args:
        folder_path: Path to folder containing PDF files
        output_dir: Directory to save extracted content
    
    Returns:
        List of extraction results
    """
    results = []
    
    # Find all PDF files
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return results
    
    print(f"Found {len(pdf_files)} PDF files\n")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        
        try:
            result = extract_pdf_content(pdf_path, output_dir)
            if result:
                results.append(result)
                print(f"  ✓ Extracted {result['total_pages']} pages")
                print(f"  ✓ Extracted {result['image_count']} images")
                print(f"  ✓ Text saved to: {result['text_file']}\n")
        except Exception as e:
            print(f"  ✗ Error processing {pdf_file}: {e}\n")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        pdf_folder = sys.argv[1]
    else:
        pdf_folder = "knowledge_base"
    
    print(f"Extracting PDFs from: {pdf_folder}")
    print("=" * 60)
    
    results = extract_all_pdfs_in_folder(pdf_folder)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if results:
        total_pages = sum(r['total_pages'] for r in results)
        total_images = sum(r['image_count'] for r in results)
        
        print(f"Processed {len(results)} PDF files")
        print(f"Total pages: {total_pages}")
        print(f"Total images extracted: {total_images}")
        print(f"\nExtracted content saved to: ./extracted_content/")
    else:
        print("No PDFs were processed")

