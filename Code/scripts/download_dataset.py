import wfdb
import os
from tqdm import tqdm

def download_mitbih():
    """Download MIT-BIH Arrhythmia Database"""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    download_dir = os.path.join(project_root, 'data', 'raw')
    os.makedirs(download_dir, exist_ok=True)
    
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
    print(f"Records: {len(records)} | Size: ~2GB")
    print(f"Destination: {download_dir}\n")
    
    original_dir = os.getcwd()
    os.chdir(download_dir)
    
    try:
        for record in tqdm(records, desc="Downloading"):
            wfdb.dl_database('mitdb', download_dir, records=[record])
    finally:
        os.chdir(original_dir)
    
    files = os.listdir(download_dir)
    print(f"\n{'='*60}")
    print(f"✓ Downloaded: {len(files)} files")
    print(f"{'='*60}")
    
    return len(files) > 0

if __name__ == "__main__":
    download_mitbih()