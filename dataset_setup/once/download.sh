#!/bin/bash
# ONCE Dataset - Google Drive API Download Commands
# Uses curl with Google Drive API access token

# Go to OAuth 2.0 Playground https://developers.google.com/oauthplayground/
# In the Select the Scope box, paste https://www.googleapis.com/auth/drive.readonly
# Click Authorize APIs and then Exchange authorization code for tokens
# Copy the Access token
# Run in terminal

# Initialize variables
OUTPUT_DIR="once_dataset"
ACCESS_TOKEN=""
CLIENT_ID="${ONCE_CLIENT_ID:?Set ONCE_CLIENT_ID}"
CLIENT_SECRET="${ONCE_CLIENT_SECRET:?Set ONCE_CLIENT_SECRET}"
REFRESH_TOKEN="${ONCE_REFRESH_TOKEN:?Set ONCE_REFRESH_TOKEN}"
declare -a failed_downloads=()
mkdir -p "$OUTPUT_DIR"

# Function to refresh the access token using the refresh token
refresh_access_token() {
  echo "Refreshing access token..."
  
  # Use curl to get a new access token
  local response=$(curl -s -X POST \
    -d "client_secret=$CLIENT_SECRET&grant_type=refresh_token&refresh_token=$REFRESH_TOKEN&client_id=$CLIENT_ID" \
    https://oauth2.googleapis.com/token)
  
  # Extract the new access token from the response
  local new_token=$(echo $response | grep -o '"access_token":"[^"]*"' | cut -d '"' -f 4)
  
  # Check if we got a new token
  if [ -z "$new_token" ]; then
    echo "ERROR: Failed to refresh access token. Response: $response"
    return 1
  fi
  
  # Update the global ACCESS_TOKEN variable
  ACCESS_TOKEN="$new_token"
  echo "Access token refreshed successfully"
  return 0
}

# Function to download a file from Google Drive using API with progress bar
download_file() {
  local file_id="$1"
  local filename="$2"
  local output_path="$OUTPUT_DIR/$filename"
  
  echo "Downloading $file_id to $output_path"
  
  # Refresh access token if it's empty
  if [ -z "$ACCESS_TOKEN" ]; then
    refresh_access_token
  fi
  
  # Use curl to download the file with progress bar
  curl -L -o "$output_path" \
       -H "Authorization: Bearer $ACCESS_TOKEN" \
       -H "Accept: application/json" \
       https://www.googleapis.com/drive/v3/files/$file_id?alt=media
}

# Store file list in an array
file_list=(
# "1Qa6JYbp7SEZtq7wI0Sw7bvhLc-CZdX8I train_infos.tar"
# "1fy_R1TOKOyvMtCgwR5YwpZtiih0c_oD5 train_lidar.tar"
# "10CajrJ02H4YEwWM87N1O1mWzYjdV18P3 train_cam09.tar"
# "1rz239PKqkFmt9m0r95AK7upjgKm1jVrP train_cam08.tar"
# "1n9t3nmy84bTJHnUOF2KUyfFOoWnyGJCi train_cam07.tar"
# "14ARNQu7-Rscr1vta-pv35z9kfYnPeV0X train_cam06.tar"
# "1P2N3uqbuFQ-ACJJv9bt8Pa15jnncrWyc train_cam05.tar"
# "1K5Y4eV4TQJiFYuBOYglVPoBxLIHjn5rA train_cam03.tar"
# "1bOVcrja8AZi8fmeZwYS98Y9mfDHU7wYl train_cam01.tar"
# "1slhW1vAR7Ps1TL-QLL4SHpMV6aN6vUse val_infos.tar"
# "1BQvbh7pdCayjoXKWPD5-_veYtBgvL_8N val_lidar.tar"
# "1Qe1UoDfTW1KaolVMXGBK9reTxgY8Ws3w val_cam09.tar"
# "1wIYm-gCK270Nk9qu99qmToXSJgZCVuNb val_cam08.tar"
# "1vVOw1dm63JbRWX0HcfmUB03k7yQqHRBQ val_cam07.tar"
# "1o5KhArGDXDdE3jQhGiOw8Vp2XsXHetc2 val_cam06.tar"
# "13o7ubCpYtt92j42KkWK4uDC7ANHs1neO val_cam05.tar"
# "1z73ARwGXSumfnzkzYU1eWVa-j2qJbkn2 val_cam03.tar"
# "1GTwLqPhj_63lC8avPR8Ul915Ct5JRbyt val_cam01.tar"
# "19R2MRZxpe-4VWJLuCQbcVTZkS4-LV1N3 test_infos.tar"
# "1-6UwiBMlGjgMtQTo3NIVFC0Il_KU_gWv test_lidar.tar"
# "1ySBHGDVusGHXmZap4M_yIz2L9-ft9t5j test_cam01.tar"
# "15l_mmAgjObYDwc2bI8B8jOm_LC-tOcob test_cam03.tar"
# "1UsyoANvqKEe96oUP3_cFAzH67I7yZc4L test_cam05.tar"
# "1b1RWkWIuw7PERBc-OV9Z4H5BTBGCnyJd test_cam06.tar"
# "1YgDsWLCxNRzfHrIMDex84Qu1oqzZhnNV test_cam07.tar"
# "16T8p0dAOArj4awOGd9udrm-9yvkauZ5U test_cam08.tar"
# "1CLf71E8RgbuMViSOFEK7eGaedr8lpikP test_cam09.tar"
# "1Bc1508pLbauPskzGMbb3See1nb4ZtyCz raw_small_infos.tar"
# "1TGfGTuYs8klpfUSJ6wTEHgUuhU32cFzb raw_lidar_p0.tar.partaa"
# "1wnAPVU_VWUyZg8iEp5S8ayWOTKOdlUSy raw_lidar_p0.tar.partab"
# "12Tier93Kcvo7ZTl2pJPK1a3WjaIBs-91 raw_lidar_p0.tar.partac"
# "1XoAbC9f3LorGCoe61PctYc-QBdJ_awwC raw_lidar_p0.tar.partad"
# "1RdPDlnvrMgz3q9GSZQI4tGZKMYvGm0CB raw_lidar_p0.tar.partae"
# "1YI0EFmCKGDxwj60MqTgLNZ4ufra1UsId raw_cam05_p0.tar"
# "1nkqZeUH8dtAKLMTHXuAj59zwmjWd3AsD raw_cam01_p0.tar"
# "1XHwYe3SZY_Hl0ZICrLRmfd4Uuy7Zg9F8 raw_cam03_p0.tar"
# "1D0tEa5Kh2yQaJePN6FCkgA26KpOPdKWT raw_cam06_p0.tar"
# "1oGf0XTgMdAEC-3_oHKpinBZk7YPEjaed raw_cam07_p0.tar"
# "1rd-wtLHQVdev52V6iMYmGhCQ1zlc6S6U raw_cam08_p0.tar"
# "100nGw44Ldocl8WvT8QAqGgCrA6-1OO_3 raw_cam09_p0.tar"
# "1zzn9RiohMEM7WNlyVC39IyRzYuPNr4e- raw_medium_infos.tar"
# "1mnVI3YJF45EkB0I1Q_qIRawC3CjMS3WN raw_lidar_p1.tar.partab"
# "1Uoj8t5j1wgg4zzdE4H3-o7k4qwIfNy_Y raw_lidar_p1.tar.partaa"
# "1UUhZmEKX6twhivZhNwWb-wT-WhIsqytN raw_lidar_p1.tar.partac"
# "1j2vFhX2eLsD6nDxHmjn_xfeKAFAxVlB- raw_lidar_p1.tar.partad"
# "1dkTt4odG4tdS-uftoOVe6xNozrbXZuPg raw_lidar_p1.tar.partae"
# "1FbaurW5guyk3LeHpj3C-yUmJ9S1pEOb1 raw_lidar_p1.tar.partaf"
# "1I44C0e1qp34VpIL6mppVQxkqQ-eoF4LL raw_lidar_p2.tar.partac"
# "1Yn4epcPn1dldb4YW6yX0WrmTZKXmzrKI raw_lidar_p2.tar.partaa"
# "15ZbD4T0Q07PHBfpIIDcH7gh5dhjqjpGL raw_lidar_p2.tar.partab"
# "10C3Y3IJ5p4nsJ2wXZOJCS3ZpM0K5xZJv raw_lidar_p2.tar.partad"
# "1vgLRANGg4FiaABB0ujvxKDQvyIPpUYNF raw_lidar_p2.tar.partae"
# "1JPYGPChVGhng7cLqQ-pW4g7M1rEltMqj raw_lidar_p2.tar.partaf"
# "1IF8S1qIE3O-FzgZzjE7BZ4TtmbJA3fDl raw_lidar_p3.tar.partac"
# "1wOtiAkJCSIwvyhGY1mim6FkmDcaBkXf9 raw_lidar_p3.tar.partaa"
# "1uuf8AwsfckVc_o9VwD_-KlXEXlduwXw_ raw_lidar_p3.tar.partab"
# "1BfX7vH0jcQD9WccjUvUrm1MEHFBayhmI raw_lidar_p3.tar.partad"
# "1nChjAxGrWyntKqE5AW0EmD99TNX6WJCv raw_lidar_p3.tar.partae"
# "1jNRrQXhrRtlr4AklvcvZ9hGkvQwhnceJ raw_lidar_p3.tar.partaf"
# "11fdA5Yk4kYbzXC4LeYonWXCBveqy0K0o raw_lidar_p4.tar.partaa"
# "1R095MNUNETzdLLxWHzjWNRIFmqIlVtCw raw_lidar_p4.tar.partab"
# "1ld0AJEd5Z4Pq9BzSp524ExdZD1od8dqL raw_lidar_p4.tar.partac"
"1dJiHbRdNoCRdxQxmgNFm5QIesklckQo- raw_lidar_p4.tar.partad"
# "1aJxrD_wWpw0TLy25cJy4OiOFR3A06XAX raw_lidar_p4.tar.partae"
# "1cddHzLXIIFBNhE-iQohvdXB3Uc2WXI7D raw_lidar_p4.tar.partaf"
# "10aVzYxgAgN9ooggj5qIEBj-AB3n-OuUR raw_cam03_p1.tar"
# "1wPTby9KSWeZ222gKtUlw4X1ZK7Be5Qqi raw_cam01_p1.tar"
# "1oTHuPkKIw_gP727UqabsktkzoOi5S_xj raw_cam05_p1.tar"
# "1toVVZBr3lp-O1pH9EY99d2cGWrcTjlhz raw_cam06_p1.tar"
# "1GyTUTjpRBLKCswFNHGqReudynSIGDvRG raw_cam07_p1.tar"
# "1suaLUBb2NnPFdbsvbeexmVl9ZiSIOYB6 raw_cam08_p1.tar"
# "1-3gspXF7SlbSN2R6WY4TxvaKDqZGOknQ raw_cam09_p1.tar"
# "1FL-aCrS57MC78evo2yH5wKqBMAaPDAQ1 raw_cam03_p2.tar"
# "1n_LB9yWKDMIOVlOkGvMfmHhnHa7jzDgi raw_cam01_p2.tar"
# "1YGuM2nxjEi7TBZPTOlauWWn2b1u2AnpR raw_cam05_p2.tar"
# "1ak9UeW-Yv0EIz4OBdRvVF4L_zBBDyoAI raw_cam06_p2.tar"
# "1zblWd5GyYcSy9PhhPpbSp2dt3_i-oF-a raw_cam07_p2.tar"
# "1zUaIvVBV5JkreF-B7JCB-hKsfkMA98rn raw_cam08_p2.tar"
# "1g1dh8aw1FHxYX8md-IL0to-g-N-9ligk raw_cam09_p2.tar"
# "1p5iYyaLs6nFCu4_CpFPUHzRcdLeX546N raw_cam05_p3.tar"
# "1iDcqz9SjD4HPUZWDni1fahZsJKhZFgS_ raw_cam01_p3.tar"
# "1OCjj2IhlRftyturvip2ylS_tmemO6KFz raw_cam03_p3.tar"
# "1fgzdvRJ_jc2MnxIQ313VNOUlp7hGKLqx raw_cam06_p3.tar"
# "1GfwW839RfiwlYRAG8PUK1LbsDTiIUlrF raw_cam07_p3.tar"
# "1wYgdBfO_bMI_JY6WOBsc-F1zMinmOxC0 raw_cam08_p3.tar"
# "1bAM7u5u3or2ojqaVUJY6_s_eB4RhVxRg raw_cam09_p3.tar"
# "1cQ9CjWtMawvJzN0OAKHfvOEgB5BgcjSm raw_cam05_p4.tar"
# "12HSmAbzh4A4YrxBNTj51_7wAtTiWKSnY raw_cam01_p4.tar"
# "18sxzuKdMnXER9IpKB50hVKscjr6CGU1h raw_cam03_p4.tar"
# "1xO5ZxXuVjZ7ANSCPNsDYvcM_TXYldTrg raw_cam06_p4.tar"
# "11xOhDXEz8JUrTVgldicKmnOD2uA-poG8 raw_cam07_p4.tar"
# "16qkQS8IEqF4KftkdOv7BhNpzz3KPtALS raw_cam08_p4.tar"
# "1VTkvKH4T05F0hbRePmwHxUxPVwjN3B5j raw_cam09_p4.tar"
# "1Pb7UVy8EURvcaPM8rIMHVTNy4xjKRNFX raw_large_infos.tar"
# "1ASx52MvepGXlu_NIOiK0ZjfWXklQAWDi raw_lidar_p5.tar.partaa"
# "1wNdm_ukOLp9s43JQpdZ3s-vMtcNzZjbW raw_lidar_p5.tar.partab"
# "1h7YnF327n8A5xem1NpgUk0oCj8PlSLI4 raw_lidar_p5.tar.partac"
# "161gsAq6vdduS8w-UHR42njjKPOHHRunb raw_lidar_p5.tar.partad"
# "1ri1mkYhbPI_kOZCEv_9hjPoNAfnY7cbp raw_lidar_p5.tar.partae"
# "1roDK4VU07Td6gmZZatb7Gd65WGYH8odp raw_lidar_p5.tar.partaf"
# "1qNF1Iyf-sEHY6SgQ2krwZxY_IYsPhM7X raw_lidar_p6.tar.partab"
# "1wc5zjC8b0pevfWqfqFW0BaXp1Y0XdU5d raw_lidar_p6.tar.partaa"
# "1MV_hMBHO9ogj7BpnXiGBluAm0eEb0h5T raw_lidar_p6.tar.partac"
# "1FSxAcvgV-mustucY7GvEr7xVslbRsonj raw_lidar_p6.tar.partad"
# "18yZBx9HoFXSxSZp1hzLBNc6fcQzTaRuI raw_lidar_p6.tar.partae"
# "1C9DUm1QyAuWWY4EDRWmH4gzhCN_9flJq raw_lidar_p6.tar.partaf"
# "1PXqgPNJ2sE0N8EwIRjo5gW41dBBN-Mja raw_lidar_p7.tar.partaa"
# "1M_BuU_ieZXUyKS598e3ff3c-vLu4zdLe raw_lidar_p7.tar.partab"
# "1dfvnzv4JLtunQlSp6xDs7EDYbsJ53o4_ raw_lidar_p7.tar.partac"
# "1cP_7o2LijyhTTo-JdchnKWwNrIa0Qcsq raw_lidar_p7.tar.partad"
# "1agvStD-HtTfCSqH7Grbpf_ceYMQGJhgc raw_lidar_p7.tar.partae"
# "1wlKi4C87Af6j1ZE6EOWxTUe2U01HH9qu raw_lidar_p7.tar.partaf"
# "1RmBJCD-oFzNVa94mVkKAvE1ygx5vqONz raw_lidar_p8.tar.partab"
# "1a0Edv7f8ah81DsIDR0uATGTsT0xRkQkc raw_lidar_p8.tar.partaa"
# "1DAemB1XYz95gBZq6uLTbgbkAWzddf9ys raw_lidar_p8.tar.partac"
# "1gKwKKrANf0nm7MkM7nlqaNVQIOFzghCM raw_lidar_p8.tar.partad"
# "1yY9RkIW_RLw7flONf5yVVsnzU3prSB2s raw_lidar_p8.tar.partae"
# "1rrlFR7BVCOOAUpzFUqbTlEFMLW4nV42w raw_lidar_p8.tar.partaf"
# "1RBz9i8_WH6f6i3pXm2-5F6wBgmdwSKTv raw_lidar_p9.tar.partad"
# "1kXsfAvL57b57j0SfZr1WdncwUstqN_3N raw_lidar_p9.tar.partaa"
# "1pqxtPiYQVUFKIgsVEdDsyyTYZMoatB9l raw_lidar_p9.tar.partab"
# "1eA_1wLyPr8s6iqlCXZngtYy3JIucJQmy raw_lidar_p9.tar.partac"
# "1paLbhgSYrDM7KtV9cvIUhAcyEOfpqEDc raw_lidar_p9.tar.partae"
# "1zNb4JY1TUtxQDYB2Zhrkg3z8UXaI_xHR raw_cam03_p5.tar"
# "1qPBYeJ-J8P0kKrtXF537w9ogtWIOEuZP raw_cam01_p5.tar"
# "11zKu77B24EXVSC0YSOhC11WDK01qflKK raw_cam05_p5.tar"
# "19JPfflBJS6OsJewBRufpoIcWhCrQ2qch raw_cam06_p5.tar"
# "1rHl1MThr3IeBs13ejGAA7MbhwLJALPw4 raw_cam07_p5.tar"
# "1yRbCmUWdZd6gfeeEveWp-buQEoKTkWTs raw_cam08_p5.tar"
# "1YO1FH5PoKQc-kteVrmytXDs9wLCHNZG0 raw_cam09_p5.tar"
# "1gG-iysR5GAZd8OEBnlgKEnYfbLethCl_ raw_cam03_p6.tar"
# "1oGmpYb6wn9pMJU8l3axsKOKLTihNUDRD raw_cam01_p6.tar"
# "1w12tJaz1syCcVcrQbdLuVtZ9StmTD_97 raw_cam05_p6.tar"
# "19gqmWiV0tIWTv8eTBxpMg_QTvgW3Xwdk raw_cam06_p6.tar"
# "1qxVli4BvfXtKcx37IxNmnQ60yVU3xYCD raw_cam07_p6.tar"
# "1LFGNXVHCYZmIogcCPgItAGBlUVgr7zND raw_cam08_p6.tar"
# "1xL8phQLn76m6pWMLo8EdxvAk8CW5RuPH raw_cam09_p6.tar"
# "1kYOLxUVL53JDDL_zBS5keUs223U840cC raw_cam03_p7.tar"
# "11-P7q0O952DJumy6Nc6PXgPjARcEgHkJ raw_cam01_p7.tar"
# "1I5m2OF7BPWoUCV9f3o1jyiuLVpL1dFxL raw_cam05_p7.tar"
# "1c3JzlymeatrSfODQ9FXBLZZY2NihreSA raw_cam06_p7.tar"
# "1NkCgHyyRMVgZ_NYR_wamgUdjXMEhoG3r raw_cam07_p7.tar"
# "1b_0ZNvK0ym8CPUE7lYo80-kK0I7YbGI1 raw_cam08_p7.tar"
# "1of2VATtall_ZGAI4ocWaaaGHqEiBDVby raw_cam09_p7.tar"
# "1foUUtSti7R4MpHkmValoRfy1hrrqwPOq raw_cam01_p8.tar"
# "1X6f2uSoLQ8iN8Mc6TU4-Aa0Q_dWKP5Vm raw_cam03_p8.tar"
# "1z2no2XfGtQVsTt5ixu6v0u11CG95yatH raw_cam05_p8.tar"
# "1v9WJsRji4d6sVNIC8SvmCnz2Ip1dthgj raw_cam06_p8.tar"
# "1JDfeyznLjvJELDND_NeABkEnsU9hnQ49 raw_cam07_p8.tar"
# "1mw21dTgDzEkqAtVZz3OTgS1dIUZgdIbR raw_cam08_p8.tar"
# "1rewg0d0pY5S69zxXKt7Jd_TFQDbhxkBs raw_cam09_p8.tar"
# "1lLt9pzT1zSqr1NAEhq4wRZK8xxnx4ztp raw_cam05_p9.tar"
# "1nTW9eo5amxLIzgTk3KempvWQURflPZ-M raw_cam01_p9.tar"
# "1nTB3NUbZugIsJ9B1jY4PP4JJqA8nvUYX raw_cam03_p9.tar"
# "1sadvkgR_B9bixsmBXSjE2kmxreL43VYF raw_cam06_p9.tar"
# "1neesKVHCZq4kVuyD25BjrMd-4l91vCIx raw_cam07_p9.tar"
# "1MWM89lVQSCQ2idv-s6wJJJw-IQFOiRHL raw_cam08_p9.tar"
# "19hXA0t7E_QyM0vWKxBTTGhWHTqf7lXMk raw_cam09_p9.tar"
)



# Function to refresh the access token using the refresh token
refresh_access_token() {
  echo "Refreshing access token..."
  
  # Use curl to get a new access token
  local response=$(curl -s -X POST \
    -d "client_secret=$CLIENT_SECRET&grant_type=refresh_token&refresh_token=$REFRESH_TOKEN&client_id=$CLIENT_ID" \
    https://oauth2.googleapis.com/token)
  
  # Extract the new access token from the response
  # local new_token=$(echo $response | grep -o '"access_token":"[^"]*"' | cut -d '"' -f 4)
  local new_token="$(printf '%s' "$response" | sed -n 's/.*"access_token"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')"
  
  echo "New access token: $new_token"
  # Check if we got a new token
  if [ -z "$new_token" ]; then
    echo "ERROR: Failed to refresh access token. Response: $response"
    return 1
  fi
  
  # Update the global ACCESS_TOKEN variable
  ACCESS_TOKEN="$new_token"
  echo "Access token refreshed successfully"
  return 0
}

# Function to download a file from Google Drive using API with progress bar
download_file() {
  local file_id="$1"
  local filename="$2"
  local output_path="$OUTPUT_DIR/$filename"
  
  echo "Downloading $file_id to $output_path"
  
  # Use curl to download the file with progress bar
  curl -L -o "$output_path" \
       -H "Authorization: Bearer $ACCESS_TOKEN" \
       -H "Accept: application/json" \
       "https://www.googleapis.com/drive/v3/files/$file_id?alt=media" \
       --progress-bar
  
  # Check if download was successful
  if [ $? -ne 0 ]; then
    echo "ERROR: Failed to download $filename (ID: $file_id)"
    failed_downloads+=("$filename (ID: $file_id)")
    return 1
  fi
  
  echo "Downloaded $filename to $output_path"
  return 0
}


# Refresh the access token before starting downloads
refresh_access_token
if [ $? -ne 0 ]; then
  echo "ERROR: Failed to get initial access token. Exiting."
  exit 1
fi

current=0
successful=0
total_files=${#file_list[@]}
echo "Starting download of $total_files files to $OUTPUT_DIR/"
echo "Progress: [0%]"
  
for entry in "${file_list[@]}"; do
  # Extract file ID and filename from the entry
  echo "Entry: $entry"
  stringarray=($entry)
  file_id="${stringarray[0]}"
  filename="${stringarray[1]}"
  
  # Download the file
  download_file "$file_id" "$filename"
  if [ $? -eq 0 ]; then
    successful=$((successful + 1))
  fi
  
  # Refresh the access token after every 3 downloads
  if [ $((current % 3)) -eq 2 ]; then
    echo "Refreshing token after 3 downloads..."
    refresh_access_token
  fi
  
  # Update progress
  current=$((current + 1))
  percent=$((current * 100 / total_files))
  echo "Progress: $percent% ($current/$total_files) - $successful successful downloads"
  
  # Add a small delay to avoid rate limiting
  sleep 0.5
done

echo -e "\nAll downloads completed!"


echo "Starting file downloads..."

echo "All files downloaded to ${OUTPUT_DIR}/ directory."
echo "You may need to extract the tar files after downloading."

# Check if any files failed to download
if [ ${#failed_downloads[@]} -gt 0 ]; then
    echo "WARNING: The following files failed to download:"
    for failed in "${failed_downloads[@]}"; do
        echo "  - $failed"
    done
    echo "You may want to retry downloading these files manually."
fi