import wfdb
import os
from tqdm import tqdm

def download_mitbih():
    """Download MIT-BIH Arrhythmia Database"""
    
    # Create directory
    download_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    os.makedirs(download_dir, exist_ok=True)
    
    # Record list
    records = [
        '100', '101', '102', '103', '104', '105', '106', '107', 
        '108', '109', '111', '112', '113', '114', '115', '116',
        '117', '118', '119', '121', '122', '123', '124', '200',
        '201', '202', '203', '205', '207', '208', '209', '210',
        '212', '213', '214', '215', '217', '219', '220', '221',
        '222', '223', '228', '230', '231', '232', '233', '234'
    ]
    
    print("="*60)
    print("MIT-BIH ARRHYTHMIA DATABASE DOWNLOAD")
    print("="*60)
    print(f"Total records: {len(records)}")
    print(f"Estimated size: ~2GB")
    print(f"Destination: {os.path.abspath(download_dir)}")
    print()
    
    successful = 0
    failed = []
    
    for record in tqdm(records, desc="Downloading"):
        try:
            # Download using correct parameter name
            wfdb.rdrecord(record, pn_dir='mitdb')
            wfdb.rdann(record, 'atr', pn_dir='mitdb')
            
            successful += 1
            
        except Exception as e:
            failed.append((record, str(e)))
            tqdm.write(f"✗ {record}: {str(e)[:60]}")
    
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)
    print(f"✓ Successful: {successful}/{len(records)}")
    if failed:
        print(f"✗ Failed: {len(failed)}")
    
    return successful == len(records)

if __name__ == "__main__":
    import sys
    success = download_mitbih()
    sys.exit(0 if success else 1)