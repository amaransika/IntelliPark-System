import pytest
from backend.main import supabase

def test_database_rls_delete_protection():
    
    print("\n[INFO] Attempting unauthorized DELETE operation on Cloud DB...")
    
    response = supabase.table("anpr_logs").delete().neq("id", 0).execute()
    
    assert len(response.data) == 0, "SECURITY BREACH: Unauthorized user deleted data!"
    
    print("[SECURE] Row Level Security (RLS) successfully blocked the deletion!")