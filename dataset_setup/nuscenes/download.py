import requests
import os
import hashlib
from tqdm import tqdm
import tarfile
import gzip
import json 
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# replace your email and password in https://www.nuscenes.org/
useremail = "[EMAIL_ADDRESS]"
password = "[PASSWORD]"

PROJECT = os.environ.get("PROJECT")
DEFAULT_DOWNLOAD_DIR = os.path.join(PROJECT,"data","nuscenes")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT,"data","nuscenes")
region = 'us' # 'us' or 'asia'


download_files = {
    "v1.0-test_meta.tgz":"b0263f5c41b780a5a10ede2da99539eb",
    "v1.0-test_blobs.tgz":"e065445b6019ecc15c70ad9d99c47b33",
    "v1.0-trainval01_blobs.tgz":"cbf32d2ea6996fc599b32f724e7ce8f2",
    "v1.0-trainval02_blobs.tgz":"aeecea4878ec3831d316b382bb2f72da",
    "v1.0-trainval03_blobs.tgz":"595c29528351060f94c935e3aaf7b995",
    "v1.0-trainval04_blobs.tgz":"b55eae9b4aa786b478858a3fc92fb72d",
    "v1.0-trainval05_blobs.tgz":"1c815ed607a11be7446dcd4ba0e71ed0",
    "v1.0-trainval06_blobs.tgz":"7273eeea36e712be290472859063a678",
    "v1.0-trainval07_blobs.tgz":"46674d2b2b852b7a857d2c9a87fc755f",
    "v1.0-trainval08_blobs.tgz":"37524bd4edee2ab99678909334313adf",
    "v1.0-trainval09_blobs.tgz":"a7fcd6d9c0934e4052005aa0b84615c0",
    "v1.0-trainval10_blobs.tgz":"31e795f2c13f62533c727119b822d739",
    "v1.0-trainval_meta.tgz":"537d3954ec34e5bcb89a35d4f6fb0d4a",
}



def login(username, password):
    headers = {
        "Content-Type": "application/x-amz-json-1.1",
        "X-Amz-Target": "AWSCognitoIdentityProviderService.InitiateAuth",
    }

    # Use json.dumps() for correct JSON formatting
    data = json.dumps({
        "AuthFlow": "USER_PASSWORD_AUTH",
        "ClientId": "7fq5jvs5ffs1c50hd3toobb3b9",
        "AuthParameters": {
            "USERNAME": username,
            "PASSWORD": password
        },
        "ClientMetadata": {}
    })

    response = requests.post(
        "https://cognito-idp.us-east-1.amazonaws.com/",
        headers=headers,
        data=data,
    )

    if response.status_code == 200:
        try:
            token = json.loads(response.content)["AuthenticationResult"]["IdToken"]
            return token
        except KeyError:
            print("Authentication failed. 'AuthenticationResult' not found in the response.")
    else:
        print("Failed to login. Status code:", response.status_code)

    return None

def download_file(url, save_file, md5, position=0):
    response = requests.get(url, stream=True)
    if save_file.endswith(".tgz"):
        content_type = response.headers.get('Content-Type', '')
        if content_type == 'application/x-tar':
            save_file = save_file.replace('.tgz', '.tar')
        elif content_type != 'application/octet-stream':
            print("unknow content type",content_type)
            return save_file

    if os.path.exists(save_file):
        print(save_file,"has downloaded")
        # check md5
        md5obj = hashlib.md5()
        with open(save_file, 'rb') as file:
            for chunk in file:
                md5obj.update(chunk)
        hash = md5obj.hexdigest()
        if hash != md5:
            print(save_file,"check md5 failed,download again")
        else:
            print(save_file,"check md5 success")
            return save_file
        
    file_size = int(response.headers.get('Content-Length', 0))
    progress_bar = tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024, desc=os.path.basename(save_file), ascii=True, position=position, leave=True)


    # save file & check md5
    md5obj = hashlib.md5()
    with open(save_file, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                md5obj.update(chunk)
                file.write(chunk)
                progress_bar.update(len(chunk))
    progress_bar.close()

    hash = md5obj.hexdigest()
    if hash != md5:
        print(save_file,"check md5 failed")
    else:
        print(save_file,"check md5 success")
    print("downloaded",save_file)

    return save_file




def check_file_needs_download(file_path, md5_expected):
    """Check if file exists and has correct MD5. Returns True if download is needed."""
    # Check for both .tgz and .tar extensions
    potential_paths = [file_path]
    if file_path.endswith(".tgz"):
        potential_paths.append(file_path.replace('.tgz', '.tar'))
    
    for path in potential_paths:
        if os.path.exists(path):
            print(f"File exists: {path}, checking MD5...")
            # Verify MD5
            md5obj = hashlib.md5()
            try:
                with open(path, 'rb') as file:
                    for chunk in file:
                        md5obj.update(chunk)
                hash_result = md5obj.hexdigest()
                if hash_result == md5_expected:
                    print(f"MD5 check passed for {path}, skipping download")
                    return False, path  # No download needed, return existing path
                else:
                    print(f"MD5 mismatch for {path}, will re-download")
            except Exception as e:
                print(f"Error checking MD5 for {path}: {e}")
    
    return True, file_path  # Download needed


def extract_tgz_to_dir(tgz_file_path, extract_dir):
    print(f"Extracting tgz {tgz_file_path} to {extract_dir}")
    os.makedirs(extract_dir, exist_ok=True)
    with gzip.open(tgz_file_path, 'rb') as f_in:
        with tarfile.open(fileobj=f_in, mode='r') as tar:
            with tqdm(unit='file', desc=f'Extracting {os.path.basename(tgz_file_path)}') as progress_bar:
                for member in tar:
                        tar.extract(member, path=extract_dir)
                        progress_bar.update(1)


def extract_tar_to_dir(tar_file_path, extract_dir):
    print(f"Extracting tar {tar_file_path} to {extract_dir}")
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(tar_file_path, 'r') as tar:
        with tqdm(unit='file', desc=f'Extracting {os.path.basename(tar_file_path)}') as progress_bar:
            for member in tar:
                    tar.extract(member, path=extract_dir)
                    progress_bar.update(1)


def download(download_dir, output_dir):
    print("Loginging...")
    bearer_token = login(useremail, password)
    headers = {
        'Authorization': f'Bearer {bearer_token}',
        'Content-Type': 'application/json',
    }

    print("Getting download urls...")
    download_data = {}
    for filename, md5 in download_files.items():
        api_url = f'https://o9k5xn5546.execute-api.us-east-1.amazonaws.com/v1/archives/v1.0/{filename}?region={region}&project=nuScenes'
        response = requests.get(api_url, headers=headers)
        if response.status_code == 200:
            print(filename, 'request success')
            download_url = response.json()['url']
            save_file = os.path.join(download_dir, filename)
            download_data[filename] = [download_url, save_file, md5]
        else:
            print(f'request failed : {response.status_code}')
            print(response.text)

    print("Checking which files need to be downloaded...")
    os.makedirs(download_dir, exist_ok=True)
    
    # Check which files need downloading
    download_tasks = []
    for output_name, (download_url, save_file, md5) in download_data.items():
        needs_download, existing_path = check_file_needs_download(save_file, md5)
        
        if needs_download:
            download_tasks.append((output_name, download_url, save_file, md5))
        else:
            # File already exists with correct MD5
            download_data[output_name] = [download_url, existing_path, md5]
    
    if download_tasks:
        print(f"Downloading {len(download_tasks)} file(s)...")
        # Download files in parallel (default 4 workers)
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_filename = {}
            for idx, (output_name, download_url, save_file, md5) in enumerate(download_tasks):
                future = executor.submit(download_file, download_url, save_file, md5, idx)
                future_to_filename[future] = output_name
            
            for future in as_completed(future_to_filename):
                output_name = future_to_filename[future]
                try:
                    actual_saved_file = future.result()
                    download_data[output_name] = [download_data[output_name][0], actual_saved_file, download_data[output_name][2]]
                except Exception as exc:
                    print(f'{output_name} generated an exception: {exc}')
    else:
        print("All files already downloaded with correct MD5, skipping download phase.")

    print("Extracting files...")
    os.makedirs(output_dir, exist_ok=True)
    for output_name, (download_url, save_file, md5) in download_data.items():
        if output_name.endswith(".tgz"):
            extract_tgz_to_dir(save_file, output_dir)
        elif output_name.endswith(".tar"):
            extract_tar_to_dir(save_file, output_dir)
        else:
            print("unknow file type", output_name)

    print("Done!")


def main(download_dir, output_dir, extract_only, pid, nproc, download_workers):
    # Ensure download_dir exists if we are going to write to it (downloading) or read from it (extract_only)
    if not extract_only: # We will be downloading
        os.makedirs(download_dir, exist_ok=True) # Create download_dir if it doesn't exist
    elif not os.path.isdir(download_dir): # Extract only, but download_dir doesn't exist or is not a directory
        print(f"Error: --download_dir '{download_dir}' does not exist or is not a directory. Cannot proceed with --extract_only.")
        return

    download_data = {} # Initialize download_data

    if not extract_only:
        print("Loginging...")
        bearer_token = login(useremail, password)
        if not bearer_token:
            print("Login failed. Cannot proceed with download.")
            return
        
        headers = {
            'Authorization': f'Bearer {bearer_token}',
            'Content-Type': 'application/json',
        }

        print("Getting download urls...")
        for filename_key, md5_val in download_files.items():
            api_url = f'https://o9k5xn5546.execute-api.us-east-1.amazonaws.com/v1/archives/v1.0/{filename_key}?region={region}&project=nuScenes'
            response = requests.get(api_url, headers=headers)
            if response.status_code == 200:
                print(filename_key,'request success')
                download_url = response.json()['url']
                save_file_path = os.path.join(download_dir, filename_key)
                download_data[filename_key] = {"url": download_url, "path": save_file_path, "md5": md5_val, "downloaded_path": None}
            else:
                print(f'Request for {filename_key} failed: {response.status_code}')
                print(response.text)
        
        if not download_data:
            print("No download URLs obtained. Aborting.")
            return

        print("Checking which files need to be downloaded...")
        download_tasks = []
        for filename_key, data in download_data.items():
            # Ensure the directory for the specific file exists
            os.makedirs(os.path.dirname(data["path"]), exist_ok=True)
            
            # Check if file needs downloading
            needs_download, existing_path = check_file_needs_download(data["path"], data["md5"])
            
            if needs_download:
                download_tasks.append((filename_key, data["url"], data["path"], data["md5"]))
            else:
                # File already exists with correct MD5
                download_data[filename_key]["downloaded_path"] = existing_path
        
        if download_tasks:
            print(f"Downloading {len(download_tasks)} file(s)...")
            # Download files in parallel
            with ThreadPoolExecutor(max_workers=download_workers) as executor:
                future_to_filename = {}
                for idx, (filename_key, url, path, md5) in enumerate(download_tasks):
                    future = executor.submit(download_file, url, path, md5, idx)
                    future_to_filename[future] = filename_key
                
                for future in as_completed(future_to_filename):
                    filename_key = future_to_filename[future]
                    try:
                        actual_saved_file = future.result()
                        download_data[filename_key]["downloaded_path"] = actual_saved_file
                    except Exception as exc:
                        print(f'{filename_key} generated an exception: {exc}')
        else:
            print("All files already downloaded with correct MD5, skipping download phase.")

    else: # extract_only is True
        print(f"Extract only mode: Searching for archives in '{download_dir}'...")
        found_any_to_extract = False
        for i, filename_key in enumerate(download_files.keys()):
            if i % nproc != pid:
                continue
            potential_tgz_path = os.path.join(download_dir, filename_key)
            potential_tar_path = os.path.join(download_dir, filename_key.replace(".tgz", ".tar") if filename_key.endswith(".tgz") else "")

            file_to_extract = None
            if os.path.exists(potential_tgz_path):
                file_to_extract = potential_tgz_path
            elif potential_tar_path and os.path.exists(potential_tar_path):
                file_to_extract = potential_tar_path
            
            if file_to_extract:
                print(f"Found: {file_to_extract}")
                download_data[filename_key] = {"url": None, "path": file_to_extract, "md5": None, "downloaded_path": file_to_extract}
                found_any_to_extract = True
            else:
                print(f"Archive for {filename_key} (or .tar version) not found in '{download_dir}'. Skipping.")
        
        if not found_any_to_extract:
            print(f"No archives found in '{download_dir}' matching expected filenames. Nothing to extract.")
            return

    if not download_data:
        print("No data to process for extraction.")
        return

    print("Extracting files...")
    os.makedirs(output_dir, exist_ok=True)


    for i, (filename_key, data) in enumerate(download_data.items()):
        file_to_extract = data.get("downloaded_path")

        if not file_to_extract or not os.path.exists(file_to_extract):
            print(f"File for {filename_key} (expected at {file_to_extract}) not available. Skipping extraction.")
            continue

        print(f"Processing extraction for: {file_to_extract}")
        if file_to_extract.endswith(".tgz"):
            extract_tgz_to_dir(file_to_extract, output_dir)
        elif file_to_extract.endswith(".tar"):
            extract_tar_to_dir(file_to_extract, output_dir)
        else:
            print(f"Unknown file type, cannot extract: {file_to_extract}")
            
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and/or extract NuScenes dataset files.")
    
    parser.add_argument(
        '--download_dir',
        type=str,
        default=DEFAULT_DOWNLOAD_DIR,
        help=f'Directory to store downloaded archives or find existing archives for extraction. (default: {DEFAULT_DOWNLOAD_DIR})'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory for extracted files. (default: {DEFAULT_OUTPUT_DIR})'
    )
    parser.add_argument(
        '--extract_only',
        action='store_true',
        help='Only extract files from --download_dir, skip downloading.'
    )
    parser.add_argument(
        '--pid',
        type=int,
        default=0,
        help='Process ID (for parallel extraction, 0-indexed)'
    )
    parser.add_argument(
        '--nproc',
        type=int,
        default=1,
        help='Total number of parallel processes for extraction'
    )
    parser.add_argument(
        '--download_workers',
        type=int,
        default=4,
        help='Number of parallel workers for downloading (default: 4)'
    )
    args = parser.parse_args()
    
    main(args.download_dir, args.output_dir, args.extract_only, args.pid, args.nproc, args.download_workers)