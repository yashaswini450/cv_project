#!/bin/bash
# package_submission.sh
# =====================
# Packages the final submission into the exact directory structure:
# <ROLL_NUMBER>/
#   solution.py
#   models/ (required, <= 250 MB total)
#   requirements.txt (required)
#   README.md (required)

set -e

# 1. Ask for roll number
read -p "Please enter your Roll Number (e.g., 210101): " roll_num

# Remove whitespace
roll_num=$(echo "$roll_num" | tr -d '[:space:]')

if [ -z "$roll_num" ]; then
    echo "❌ Error: Roll Number cannot be empty!"
    exit 1
fi

echo "📦 Packaging submission into folder: $roll_num/ ..."

# 2. Setup temporary folder named after roll number
rm -rf "$roll_num"
mkdir -p "$roll_num"

# 3. Copy files from final_submission
cp final_submission/solution.py "$roll_num/"
cp final_submission/requirements.txt "$roll_num/"
cp final_submission/README.md "$roll_num/"
cp final_submission/final_submission_report.md "$roll_num/report.md"
cp -R final_submission/src "$roll_num/"
cp -R final_submission/models "$roll_num/"

# 4. Create ZIP archive
zip_name="${roll_num}.zip"
rm -f "$zip_name"
echo "🗜️ Creating ZIP archive: $zip_name ..."
zip -r "$zip_name" "$roll_num"

# 5. Clean up temporary directory
rm -rf "$roll_num"

echo "=========================================================="
echo "✅ SUCCESS! Final submission packaged into: $zip_name"
echo "📏 Total ZIP Size: $(du -sh $zip_name | cut -f1)"
echo "🚀 You are ready to upload $zip_name!"
echo "=========================================================="
