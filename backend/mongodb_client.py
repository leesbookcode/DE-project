"""
MongoDB Connection Client
"""

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import config


def get_mongodb_client():
    """
    Get MongoDB client instance
    
    Returns:
        MongoClient: Connected MongoDB client
        
    Usage:
        client = get_mongodb_client()
        db = client["de-project"]
        collection = db["my_collection"]
        collection.insert_one({"name": "test"})
    """
    try:
        client = MongoClient(
            config.MONGODB_URI,
            serverSelectionTimeoutMS=5000
        )
        
        # Test connection
        client.admin.command('ping')
        print(f"✓ Connected to MongoDB")
        
        return client
        
    except ConnectionFailure as e:
        print(f"✗ MongoDB connection failed: {e}")
        raise
    except Exception as e:
        print(f"✗ Error: {e}")
        raise


if __name__ == "__main__":
    # Test connection
    client = get_mongodb_client()
    
    # Get database
    db = client[config.MONGODB_DATABASE]
    
    # Test basic operation
    collection = db["test_collection"]
    result = collection.insert_one({"name": "Test Document", "value": 123})
    print(f"Inserted document ID: {result.inserted_id}")
    
    # Find document
    doc = collection.find_one({"_id": result.inserted_id})
    print(f"Found document: {doc}")
    
    client.close()
    print("✓ Connection closed")

