import pytest
from backend.main import supabase

def test_supabase_cloud_connection():
    assert supabase is not None, "Supabase connection failed! Check URL and Key."
    
    try:
        response = supabase.table("anpr_logs").select("*").limit(1).execute()
        assert type(response.data) is list
    except Exception as e:
        pytest.fail(f"Cloud DB Integration Error: {e}")