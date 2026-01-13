#!/usr/bin/env python3
"""
Run Supabase migrations via direct PostgreSQL connection.

Usage:
    1. Get your database connection string from Supabase Dashboard:
       Project Settings → Database → Connection string → URI
    
    2. Set it as environment variable:
       export DATABASE_URL="postgresql://postgres:[YOUR-PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres"
    
    3. Run this script:
       python scripts/run_migration.py src/backend/migrations/008_webhooks.sql

Or provide the connection string as argument:
    python scripts/run_migration.py src/backend/migrations/008_webhooks.sql --db-url "postgresql://..."
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Run Supabase migration")
    parser.add_argument("migration_file", help="Path to SQL migration file")
    parser.add_argument("--db-url", help="PostgreSQL connection URL (or set DATABASE_URL env)")
    args = parser.parse_args()
    
    # Get database URL
    db_url = args.db_url or os.getenv("DATABASE_URL")
    
    if not db_url:
        print("❌ No database connection URL provided!")
        print("\nTo run migrations, you need the PostgreSQL connection string from Supabase:")
        print("  1. Go to: https://supabase.com/dashboard/project/nsbjwhxdkxnyggtuxjjp/settings/database")
        print("  2. Copy the 'Connection string' → 'URI' value")
        print("  3. Run: export DATABASE_URL='postgresql://...'")
        print("  4. Then run this script again")
        print("\nOr run the SQL directly in Supabase SQL Editor:")
        print("  https://supabase.com/dashboard/project/nsbjwhxdkxnyggtuxjjp/sql/new")
        sys.exit(1)
    
    # Read migration file
    if not os.path.exists(args.migration_file):
        print(f"❌ Migration file not found: {args.migration_file}")
        sys.exit(1)
    
    with open(args.migration_file, "r") as f:
        sql = f.read()
    
    print(f"📄 Migration file: {args.migration_file}")
    print(f"📊 SQL size: {len(sql)} bytes")
    
    try:
        import psycopg2
    except ImportError:
        print("❌ psycopg2 not installed. Installing...")
        os.system(f"{sys.executable} -m pip install psycopg2-binary")
        import psycopg2
    
    try:
        print(f"🔌 Connecting to database...")
        conn = psycopg2.connect(db_url)
        conn.autocommit = True
        cursor = conn.cursor()
        
        print(f"🚀 Running migration...")
        cursor.execute(sql)
        
        print("✅ Migration completed successfully!")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
