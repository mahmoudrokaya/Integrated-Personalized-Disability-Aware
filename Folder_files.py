from pathlib import Path

# Target folder
ROOT_DIR = Path(
    r"D:\47\472\New-Papers\Integrated Personalized Disability-Aware\Experiments"
)

print(f"\nFolder structure for:\n{ROOT_DIR}\n")
print("=" * 80)

def print_tree(path, indent=""):
    items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))

    for item in items:
        if item.is_dir():
            print(f"{indent}[DIR]  {item.name}")
            print_tree(item, indent + "    ")
        else:
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"{indent}[FILE] {item.name}  ({size_mb:.2f} MB)")

if ROOT_DIR.exists():
    print_tree(ROOT_DIR)
else:
    print("Folder does not exist.")