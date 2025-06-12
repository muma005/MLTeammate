import os
import shutil

# Define the folders to move into the new ml_teammate package
folders_to_move = [
    "automl", "learners", "search", "data", "utils", "interface", "experiments"
]

# Step 1: Create ml_teammate directory if it doesn't exist
if not os.path.exists("ml_teammate"):
    os.makedirs("ml_teammate")
    print("✅ Created 'ml_teammate/' package directory.")

# Step 2: Move each folder into ml_teammate
for folder in folders_to_move:
    if os.path.exists(folder):
        dest = os.path.join("ml_teammate", folder)
        if not os.path.exists(dest):
            shutil.move(folder, dest)
            print(f"✅ Moved '{folder}' → 'ml_teammate/{folder}'")
        else:
            print(f"⚠️ Skipped: 'ml_teammate/{folder}' already exists.")
    else:
        print(f"❌ Folder '{folder}' not found — skipping.")

# Step 3: Ensure __init__.py exists in all moved folders
for root, dirs, files in os.walk("ml_teammate"):
    if "__init__.py" not in files:
        open(os.path.join(root, "__init__.py"), "a").close()
        print(f"📦 Added '__init__.py' to '{root}'")

print("\n🎉 Done restructuring! You can now run: pip install -e .")

