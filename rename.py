import os
import glob


def rename_files(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.png"))

    for file in files:
        filename = os.path.basename(file)
        name, ext = os.path.splitext(filename)

        if "_pixels0" in name:
            new_name = name.replace("_pixels0", "") + ext
            new_file = os.path.join(folder_path, new_name)

            os.rename(file, new_file)
            print(f"Renamed: {filename} -> {new_name}")


if __name__ == "__main__":
    folder_path = r"./datasets/NUAA-SIRST/masks"
    rename_files(folder_path)