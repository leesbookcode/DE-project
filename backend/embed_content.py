"""
Embed extracted content (text and images) using CLIP model and store in MongoDB
"""

import os
from pathlib import Path
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from mongodb_client import get_mongodb_client
import config


# Set cache directory to project folder to avoid permission issues
cache_dir = os.path.join(os.path.dirname(__file__), ".cache")
os.makedirs(cache_dir, exist_ok=True)

# Set Hugging Face environment variables to use local cache
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir
os.environ['HF_DATASETS_CACHE'] = cache_dir
os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir

# Load CLIP model and processor
print("Loading CLIP model (first time will download ~600MB)...")
print(f"Cache directory: {cache_dir}")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=cache_dir)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=cache_dir)
print("✓ CLIP model loaded\n")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100):
    """
    Split text into chunks with overlap
    CLIP has max token limit, so use smaller chunks
    
    Args:
        text: Text to split
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks
    
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        
        if chunk.strip():
            chunks.append(chunk.strip())
        
        start += chunk_size - overlap
    
    return chunks


def get_text_embedding(text: str):
    """Get embedding for text using CLIP"""
    try:
        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
        
        # Normalize embedding
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Convert to list
        embedding = text_features[0].cpu().numpy().tolist()
        
        return embedding
        
    except Exception as e:
        print(f"  Error getting text embedding: {e}")
        return None


def get_image_embedding(image_path: str):
    """Get embedding for image using CLIP"""
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Process image
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        
        # Normalize embedding
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Convert to list
        embedding = image_features[0].cpu().numpy().tolist()
        
        return embedding
        
    except Exception as e:
        print(f"  Error getting image embedding: {e}")
        return None


def process_extracted_content(content_dir: str = "extracted_content"):
    """Process all extracted content and store embeddings in MongoDB"""
    
    # Connect to MongoDB
    mongo_client = get_mongodb_client()
    db = mongo_client[config.MONGODB_DATABASE]
    embeddings_collection = db["embeddings"]
    
    # Clear existing embeddings (optional)
    embeddings_collection.delete_many({})
    print("Cleared existing embeddings\n")
    
    content_path = Path(content_dir)
    
    if not content_path.exists():
        print(f"Content directory not found: {content_dir}")
        return
    
    total_text_chunks = 0
    total_images = 0
    
    # Process each document folder
    for doc_folder in content_path.iterdir():
        if not doc_folder.is_dir():
            continue
        
        doc_name = doc_folder.name
        print(f"Processing: {doc_name}")
        print("=" * 60)
        
        # Process text file
        text_file = doc_folder / f"{doc_name}_text.txt"
        if text_file.exists():
            print(f"  Processing text...")
            
            with open(text_file, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            # Chunk text
            chunks = chunk_text(text_content, chunk_size=500, overlap=100)
            print(f"  Created {len(chunks)} text chunks")
            
            # Embed each chunk
            for idx, chunk in enumerate(chunks):
                embedding = get_text_embedding(chunk)
                
                if embedding:
                    # Store in MongoDB
                    embeddings_collection.insert_one({
                        "document": doc_name,
                        "type": "text",
                        "chunk_index": idx,
                        "content": chunk,
                        "embedding": embedding,
                        "source_file": str(text_file)
                    })
                    total_text_chunks += 1
                    
                    if (idx + 1) % 10 == 0:
                        print(f"    Embedded {idx + 1}/{len(chunks)} chunks...")
            
            print(f"  ✓ Embedded {len(chunks)} text chunks")
        
        # Process images
        images_dir = doc_folder / "images"
        if images_dir.exists():
            image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
            
            if image_files:
                print(f"  Processing {len(image_files)} images...")
                
                for idx, img_file in enumerate(image_files):
                    embedding = get_image_embedding(str(img_file))
                    
                    if embedding:
                        # Store in MongoDB
                        embeddings_collection.insert_one({
                            "document": doc_name,
                            "type": "image",
                            "image_name": img_file.name,
                            "embedding": embedding,
                            "source_file": str(img_file)
                        })
                        total_images += 1
                
                print(f"  ✓ Embedded {len(image_files)} images")
        
        print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total text chunks embedded: {total_text_chunks}")
    print(f"Total images embedded: {total_images}")
    print(f"Total embeddings stored: {total_text_chunks + total_images}")
    
    # Create index for faster similarity search
    print("\nCreating vector search index...")
    try:
        embeddings_collection.create_index("document")
        embeddings_collection.create_index("type")
        print("✓ Indexes created")
    except Exception as e:
        print(f"Warning: Could not create indexes: {e}")
    
    mongo_client.close()
    print("\n✓ Done!")


if __name__ == "__main__":
    process_extracted_content()

