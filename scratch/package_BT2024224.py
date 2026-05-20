import os
import zipfile
import shutil

def package():
    roll_num = "BT2024224"
    src_dir = "/Users/yashaswini/cv_project/AID728"
    output_dir = "/Users/yashaswini/cv_project"
    
    code_zip_path = os.path.join(output_dir, f"{roll_num}_code.zip")
    models_zip_path = os.path.join(output_dir, f"{roll_num}_models.zip")
    
    print("📦 Starting split packaging for roll number:", roll_num)
    
    # 1. Create BT2024224_code.zip
    print("🗜️ Creating code zip:", code_zip_path)
    with zipfile.ZipFile(code_zip_path, 'w', zipfile.ZIP_DEFLATED) as code_zip:
        # Files at the root of the folder
        root_files = ['solution.py', 'requirements.txt', 'README.md']
        for rf in root_files:
            file_path = os.path.join(src_dir, rf)
            if os.path.exists(file_path):
                arcname = os.path.join(roll_num, rf)
                code_zip.write(file_path, arcname)
                print(f"  Added code file: {rf} -> {arcname}")
                
        # Rename final_submission_report.md to report.md
        report_path = os.path.join(src_dir, 'final_submission_report.md')
        if os.path.exists(report_path):
            arcname = os.path.join(roll_num, 'report.md')
            code_zip.write(report_path, arcname)
            print(f"  Added report: final_submission_report.md -> {arcname}")
            
        # Copy src/ directory
        src_path = os.path.join(src_dir, 'src')
        if os.path.exists(src_path):
            for root, dirs, files in os.walk(src_path):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, src_dir)
                    arcname = os.path.join(roll_num, rel_path)
                    code_zip.write(full_path, arcname)
            print("  Added src/ directory recursively.")
            
        # Copy sample_images/ directory
        img_path = os.path.join(src_dir, 'sample_images')
        if os.path.exists(img_path):
            for root, dirs, files in os.walk(img_path):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, src_dir)
                    arcname = os.path.join(roll_num, rel_path)
                    code_zip.write(full_path, arcname)
            print("  Added sample_images/ directory recursively.")

    # 2. Create BT2024224_models.zip
    print("🗜️ Creating models zip:", models_zip_path)
    with zipfile.ZipFile(models_zip_path, 'w', zipfile.ZIP_DEFLATED) as models_zip:
        models_path = os.path.join(src_dir, 'models')
        if os.path.exists(models_path):
            for root, dirs, files in os.walk(models_path):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, src_dir)
                    # We store it inside BT2024224/models/... so they merge perfectly!
                    arcname = os.path.join(roll_num, rel_path)
                    models_zip.write(full_path, arcname)
            print("  Added models/ directory recursively.")

    # 3. Handle PDF Report
    pdf_report_src = "/Users/yashaswini/cv_project/AID728_report.pdf"
    pdf_report_dst = os.path.join(output_dir, f"{roll_num}_report.pdf")
    if os.path.exists(pdf_report_src):
        shutil.copy(pdf_report_src, pdf_report_dst)
        print(f"📄 Copied PDF report: {pdf_report_src} -> {pdf_report_dst}")
    else:
        print("⚠️ Warning: AID728_report.pdf not found in workspace root!")

    print("==========================================================")
    print("✅ Split packaging complete!")
    print("📏 Code ZIP Size:", os.path.getsize(code_zip_path) // 1024, "KB")
    print("📏 Models ZIP Size:", os.path.getsize(models_zip_path) // (1024*1024), "MB")
    print("==========================================================")

if __name__ == '__main__':
    package()
