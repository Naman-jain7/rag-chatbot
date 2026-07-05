-- Migration: add_audio_support
-- Adds source_type (document | audio) and extra_metadata JSONB to user_documents.
-- document_chunks needs no changes — all existing columns map cleanly to audio chunks.

ALTER TABLE user_documents
    ADD COLUMN IF NOT EXISTS source_type   TEXT  NOT NULL DEFAULT 'document',
    ADD COLUMN IF NOT EXISTS extra_metadata JSONB;

-- Optional index for fast type-filtered listing queries
CREATE INDEX IF NOT EXISTS idx_user_documents_source_type
    ON user_documents (user_id, source_type);

COMMENT ON COLUMN user_documents.source_type IS
    'Type of uploaded content: ''document'' for text/PDF files, ''audio'' for MP3/WAV/etc.';

COMMENT ON COLUMN user_documents.extra_metadata IS
    'Arbitrary JSONB bag for file-type-specific metadata.
     For audio: { "duration": 325.4, "timestamps": [{"start": 0.0, "end": 30.0, "text": "..."}, ...] }';
