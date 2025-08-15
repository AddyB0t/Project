#!/usr/bin/env python3
"""
Test script for UUID assignment functionality
"""
import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.embedding_service import EmbeddingService

async def test_uuid_assignment():
    """Test the direct UUID assignment functionality"""
    try:
        print("🧪 STARTING UUID ASSIGNMENT TEST")
        print("=" * 50)
        
        # Initialize embedding service
        print("📋 Initializing embedding service...")
        embedding_service = EmbeddingService()
        
        # Test the direct UUID assignment
        print("🎲 Testing direct UUID assignment...")
        result = await embedding_service.assign_random_uuids_directly()
        
        # Display results
        print("\n📊 TEST RESULTS:")
        print("=" * 30)
        print(f"Success: {result['success']}")
        print(f"Message: {result['message']}")
        
        if result['success']:
            print(f"Documents processed: {result['documents_processed']}")
            print(f"Successful documents: {result['successful_documents']}")
            print(f"Failed documents: {result['failed_documents']}")
            print(f"Total embeddings updated: {result['total_embeddings_updated']}")
            print(f"Total metadata updated: {result['total_metadata_updated']}")
            
            if result['successful_docs']:
                print("\n✅ SUCCESSFUL UPDATES:")
                for doc in result['successful_docs']:
                    doc_result = result['results'][doc]
                    print(f"   📄 {doc}")
                    print(f"      Old ID: {doc_result['old_id']}")
                    print(f"      New UUID: {doc_result['new_uuid']}")
                    print(f"      Embeddings updated: {doc_result['embeddings_updated']}")
                    print(f"      Metadata updated: {doc_result['metadata_updated']}")
            
            if result['failed_docs']:
                print("\n❌ FAILED UPDATES:")
                for doc in result['failed_docs']:
                    doc_result = result['results'][doc]
                    print(f"   📄 {doc}")
                    print(f"      Error: {doc_result.get('error', 'Unknown error')}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
        
        print("\n🧪 TEST COMPLETED")
        return result
        
    except Exception as e:
        print(f"❌ TEST FAILED: {str(e)}")
        return None

if __name__ == "__main__":
    # Run the test
    result = asyncio.run(test_uuid_assignment())
    
    # Exit with appropriate code
    if result and result['success']:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Tests failed!")
        sys.exit(1)