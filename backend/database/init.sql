# PostgreSQL Initialization Script
-- Ensures default database and extensions are ready
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Log initialization
DO $$
BEGIN
    RAISE NOTICE 'ModelOps Hub Database Initialized Successfully.';
END $$;
