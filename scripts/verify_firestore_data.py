import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.getcwd())

from src.repositories.firestore_repo import FirestoreRepository
import src.cloud_config as cloud_config

def verify_indices_and_data():
    repo = FirestoreRepository()
    print(f"--- FIRESTORE DIAGNOSTIC (Project: {cloud_config.PROJECT_ID}) ---")
    
    # 1. Check Raw Count
    try:
        all_snaps = repo.db.collection('snapshots').limit(10).get()
        print(f"Total Recent Snapshots: {len(all_snaps)}")
        
        has_manual = False
        for doc in all_snaps:
            d = doc.to_dict()
            t_type = d.get('trigger_type', 'N/A')
            print(f"Snapshot ID: {doc.id} | Trigger: {t_type} | Time: {d.get('timestamp')}")
            if t_type == 'MANUAL_REFRESH':
                has_manual = True
                
        if not has_manual:
            print("\n[WARNING] No snapshots with 'MANUAL_REFRESH' tag found in the last 10 records.")
            print("Try clicking 'Force Market Refresh' again and wait 5 seconds.")
            
    except Exception as e:
        print(f"\n[ERROR] General Query Failed: {e}")

    # 2. Test the specific HUD query (Checks for index requirements)
    print("\n--- Testing HUD Baseline Query ---")
    try:
        manual = repo.get_latest_manual_snapshot()
        if manual:
            print(f"SUCCESS: Found latest manual snapshot from {manual.get('timestamp')}")
        else:
            print("INFO: HUD baseline query returned None (No manual records found yet)")
    except Exception as e:
        if "index" in str(e).lower():
            print("\n[CRITICAL] INDEX REQUIRED: Firestore needs a composite index.")
            print("Please check your terminal for a Google Cloud Console link to create it.")
        else:
            print(f"\n[ERROR] HUD query failed: {e}")

if __name__ == "__main__":
    verify_indices_and_data()
