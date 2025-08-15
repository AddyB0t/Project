-- Supabase Database Schema for Document Embeddings
-- Enable the pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Table 1: Main embeddings table
CREATE TABLE IF NOT EXISTS document_embeddings (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    document_id VARCHAR(255) NOT NULL,
    document_name VARCHAR(500) NOT NULL,
    chunk_id VARCHAR(255) NOT NULL UNIQUE,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    embedding VECTOR(768) NOT NULL, -- 768 dimensions for all-mpnet-base-v2
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);


-- Table 3: Similarity searches analytics table
CREATE TABLE IF NOT EXISTS similarity_searches (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    query_text TEXT NOT NULL,
    query_embedding VECTOR(768) NOT NULL,
    results_found INTEGER DEFAULT 0,
    search_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_document_embeddings_document_id ON document_embeddings(document_id);
CREATE INDEX IF NOT EXISTS idx_document_embeddings_document_name ON document_embeddings(document_name);
CREATE INDEX IF NOT EXISTS idx_document_embeddings_chunk_id ON document_embeddings(chunk_id);
CREATE INDEX IF NOT EXISTS idx_document_embeddings_embedding ON document_embeddings 
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);


CREATE INDEX IF NOT EXISTS idx_similarity_searches_timestamp ON similarity_searches(search_timestamp);
CREATE INDEX IF NOT EXISTS idx_similarity_searches_query ON similarity_searches USING gin(to_tsvector('english', query_text));

-- Function for similarity search using cosine similarity
CREATE OR REPLACE FUNCTION search_similar_embeddings(
    query_embedding VECTOR(768),
    similarity_threshold FLOAT DEFAULT 0.5,
    match_count INT DEFAULT 5
)
RETURNS TABLE (
    document_id VARCHAR,
    document_name VARCHAR,
    chunk_id VARCHAR,
    chunk_text TEXT,
    chunk_index INTEGER,
    similarity_score FLOAT,
    original_filename VARCHAR,
    file_type VARCHAR,
    metadata JSONB
)
LANGUAGE SQL STABLE
AS $$
    SELECT 
        de.document_id,
        de.document_name,
        de.chunk_id,
        de.chunk_text,
        de.chunk_index,
        1 - (de.embedding <=> query_embedding) AS similarity_score,
        dm.original_filename,
        dm.file_type,
        de.metadata
    FROM document_embeddings de
    JOIN documents_metadata dm ON de.document_id = dm.document_id
    WHERE 1 - (de.embedding <=> query_embedding) > similarity_threshold
    ORDER BY de.embedding <=> query_embedding ASC
    LIMIT match_count;
$$;

-- Function for similarity search with document name filtering
CREATE OR REPLACE FUNCTION search_similar_embeddings_by_name(
    query_embedding VECTOR(768),
    similarity_threshold FLOAT DEFAULT 0.5,
    match_count INT DEFAULT 5,
    filter_document_name VARCHAR DEFAULT NULL
)
RETURNS TABLE (
    document_id VARCHAR,
    document_name VARCHAR,
    chunk_id VARCHAR,
    chunk_text TEXT,
    chunk_index INTEGER,
    similarity_score FLOAT,
    metadata JSONB
)
LANGUAGE SQL STABLE
AS $$
    SELECT 
        de.document_id,
        de.document_name,
        de.chunk_id,
        de.chunk_text,
        de.chunk_index,
        1 - (de.embedding <=> query_embedding) AS similarity_score,
        de.metadata
    FROM document_embeddings de
    WHERE 
        1 - (de.embedding <=> query_embedding) > similarity_threshold
        AND (filter_document_name IS NULL OR de.document_name = filter_document_name)
    ORDER BY de.embedding <=> query_embedding ASC
    LIMIT match_count;
$$;

-- Function to get document statistics
CREATE OR REPLACE FUNCTION get_embedding_statistics()
RETURNS TABLE (
    total_documents BIGINT,
    total_chunks BIGINT,
    total_characters BIGINT,
    avg_chunks_per_document NUMERIC,
    avg_characters_per_document NUMERIC
)
LANGUAGE SQL STABLE
AS $$
    SELECT 
        COUNT(DISTINCT dm.document_id) as total_documents,
        COUNT(de.id) as total_chunks,
        SUM(dm.total_characters) as total_characters,
        ROUND(COUNT(de.id)::NUMERIC / NULLIF(COUNT(DISTINCT dm.document_id), 0), 2) as avg_chunks_per_document,
        ROUND(SUM(dm.total_characters)::NUMERIC / NULLIF(COUNT(DISTINCT dm.document_id), 0), 2) as avg_characters_per_document
    FROM documents_metadata dm
    LEFT JOIN document_embeddings de ON dm.document_id = de.document_id;
$$;

-- Function to get available document names for selection
CREATE OR REPLACE FUNCTION get_available_document_names()
RETURNS TABLE (
    document_name VARCHAR,
    document_count BIGINT
)
LANGUAGE SQL STABLE
AS $$
    SELECT 
        de.document_name,
        COUNT(*) as document_count
    FROM document_embeddings de
    WHERE de.document_name IS NOT NULL 
      AND de.document_name != ''
      AND LENGTH(TRIM(de.document_name)) > 0
    GROUP BY de.document_name
    ORDER BY de.document_name;
$$;

-- Function to clean up old embeddings (optional)
CREATE OR REPLACE FUNCTION cleanup_old_embeddings(days_old INTEGER DEFAULT 30)
RETURNS INTEGER
LANGUAGE PLPGSQL
AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    -- Delete old similarity searches
    DELETE FROM similarity_searches 
    WHERE search_timestamp < NOW() - INTERVAL '1 day' * days_old;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    RETURN deleted_count;
END;
$$;

-- Create trigger to automatically update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply the trigger to tables that need it
CREATE TRIGGER update_document_embeddings_updated_at 
    BEFORE UPDATE ON document_embeddings 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_documents_metadata_updated_at 
    BEFORE UPDATE ON documents_metadata 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- RLS (Row Level Security) policies (optional - uncomment if needed)
-- ALTER TABLE document_embeddings ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE documents_metadata ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE similarity_searches ENABLE ROW LEVEL SECURITY;

-- Example RLS policies (uncomment and modify as needed)
-- CREATE POLICY "Users can view their own document embeddings" ON document_embeddings
--     FOR SELECT USING (auth.uid()::text = (metadata->>'user_id'));

-- CREATE POLICY "Users can insert their own document embeddings" ON document_embeddings
--     FOR INSERT WITH CHECK (auth.uid()::text = (metadata->>'user_id'));

-- Grant necessary permissions (adjust as needed for your use case)
-- GRANT USAGE ON SCHEMA public TO anon, authenticated;
-- GRANT ALL ON ALL TABLES IN SCHEMA public TO anon, authenticated;
-- GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO anon, authenticated;
-- GRANT ALL ON ALL FUNCTIONS IN SCHEMA public TO anon, authenticated;

-- Insert some sample data for testing (optional)
-- INSERT INTO documents_metadata (document_id, original_filename, file_type, file_size, total_chunks, total_characters)
-- VALUES 
--     ('test-doc-1', 'sample.pdf', 'pdf', 1024, 3, 2500),
--     ('test-doc-2', 'example.docx', 'docx', 2048, 5, 4000);

COMMENT ON TABLE document_embeddings IS 'Stores text chunks and their vector embeddings for semantic search';
COMMENT ON TABLE documents_metadata IS 'Stores metadata about processed documents';
COMMENT ON TABLE similarity_searches IS 'Analytics table for tracking similarity search queries';
COMMENT ON FUNCTION search_similar_embeddings IS 'Performs vector similarity search using cosine distance';
COMMENT ON FUNCTION get_embedding_statistics IS 'Returns aggregate statistics about stored embeddings';
COMMENT ON FUNCTION cleanup_old_embeddings IS 'Removes old similarity search records to manage storage';

-- ========================================================================
-- USER AUTHENTICATION AND DOCUMENT ISOLATION IMPLEMENTATION
-- ========================================================================

-- Table 4: Users authentication table
CREATE TABLE IF NOT EXISTS users (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(255),
    last_name VARCHAR(255),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add user_id column to existing tables
ALTER TABLE documents_metadata 
ADD COLUMN IF NOT EXISTS user_id VARCHAR(255);

ALTER TABLE document_embeddings 
ADD COLUMN IF NOT EXISTS user_id VARCHAR(255);

ALTER TABLE similarity_searches 
ADD COLUMN IF NOT EXISTS user_id VARCHAR(255);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_user_id ON users(user_id);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active);

CREATE INDEX IF NOT EXISTS idx_documents_metadata_user_id ON documents_metadata(user_id);
CREATE INDEX IF NOT EXISTS idx_document_embeddings_user_id ON document_embeddings(user_id);
CREATE INDEX IF NOT EXISTS idx_similarity_searches_user_id ON similarity_searches(user_id);

-- Add foreign key constraints for data integrity
ALTER TABLE documents_metadata 
ADD CONSTRAINT fk_documents_metadata_user_id 
FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE;

ALTER TABLE document_embeddings 
ADD CONSTRAINT fk_document_embeddings_user_id 
FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE;

ALTER TABLE similarity_searches 
ADD CONSTRAINT fk_similarity_searches_user_id 
FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE;

-- Enable Row Level Security on all tables
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents_metadata ENABLE ROW LEVEL SECURITY;
ALTER TABLE similarity_searches ENABLE ROW LEVEL SECURITY;

-- RLS Policies for users table
CREATE POLICY "Users can view their own profile" ON users
    FOR SELECT USING (auth.uid()::text = user_id);

CREATE POLICY "Users can update their own profile" ON users
    FOR UPDATE USING (auth.uid()::text = user_id);

-- RLS Policies for documents_metadata
CREATE POLICY "Users can view their own documents metadata" ON documents_metadata
    FOR SELECT USING (auth.uid()::text = user_id);

CREATE POLICY "Users can insert their own documents metadata" ON documents_metadata
    FOR INSERT WITH CHECK (auth.uid()::text = user_id);

CREATE POLICY "Users can update their own documents metadata" ON documents_metadata
    FOR UPDATE USING (auth.uid()::text = user_id);

CREATE POLICY "Users can delete their own documents metadata" ON documents_metadata
    FOR DELETE USING (auth.uid()::text = user_id);

-- RLS Policies for document_embeddings
CREATE POLICY "Users can view their own document embeddings" ON document_embeddings
    FOR SELECT USING (auth.uid()::text = user_id);

CREATE POLICY "Users can insert their own document embeddings" ON document_embeddings
    FOR INSERT WITH CHECK (auth.uid()::text = user_id);

CREATE POLICY "Users can delete their own document embeddings" ON document_embeddings
    FOR DELETE USING (auth.uid()::text = user_id);

-- RLS Policies for similarity_searches
CREATE POLICY "Users can view their own similarity searches" ON similarity_searches
    FOR SELECT USING (auth.uid()::text = user_id);

CREATE POLICY "Users can insert their own similarity searches" ON similarity_searches
    FOR INSERT WITH CHECK (auth.uid()::text = user_id);

-- Updated similarity search function with user filtering
CREATE OR REPLACE FUNCTION search_similar_embeddings_by_user(
    query_embedding VECTOR(768),
    user_id_filter VARCHAR(255),
    similarity_threshold FLOAT DEFAULT 0.5,
    match_count INT DEFAULT 5
)
RETURNS TABLE (
    document_id VARCHAR,
    document_name VARCHAR,
    chunk_id VARCHAR,
    chunk_text TEXT,
    chunk_index INTEGER,
    similarity_score FLOAT,
    original_filename VARCHAR,
    file_type VARCHAR,
    metadata JSONB
)
LANGUAGE SQL STABLE
AS $$
    SELECT 
        de.document_id,
        de.document_name,
        de.chunk_id,
        de.chunk_text,
        de.chunk_index,
        1 - (de.embedding <=> query_embedding) AS similarity_score,
        dm.original_filename,
        dm.file_type,
        de.metadata
    FROM document_embeddings de
    JOIN documents_metadata dm ON de.document_id = dm.document_id
    WHERE 
        1 - (de.embedding <=> query_embedding) > similarity_threshold
        AND de.user_id = user_id_filter
        AND dm.user_id = user_id_filter
    ORDER BY de.embedding <=> query_embedding ASC
    LIMIT match_count;
$$;

-- Apply the updated_at trigger to users table
CREATE TRIGGER update_users_updated_at 
    BEFORE UPDATE ON users 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant necessary permissions for authenticated users
GRANT USAGE ON SCHEMA public TO authenticated;
GRANT ALL ON users TO authenticated;
GRANT ALL ON documents_metadata TO authenticated;
GRANT ALL ON document_embeddings TO authenticated;
GRANT ALL ON similarity_searches TO authenticated;

-- Comments for new table and function
COMMENT ON TABLE users IS 'Stores user authentication and profile information';
COMMENT ON FUNCTION search_similar_embeddings_by_user IS 'Performs user-filtered vector similarity search using cosine distance';