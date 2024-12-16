import os
import shutil
from tqdm import tqdm

# FoR
def filter_for(dst_root, count):
    src_root = "/DETECT/FoR"
    for folder in tqdm(os.listdir(src_root)): # for-2sec
        folder_path = os.path.join(src_root, folder)
        for sub_folder in tqdm(os.listdir(folder_path)): # for-2seconds
            sub_folder_path = os.path.join(folder_path, sub_folder)
            for i, dt in enumerate(['training', 'validation','testing']):
                for j, fr in enumerate(["real", "fake"]):
                    src_folder_path = os.path.join(sub_folder_path, dt, fr)
                    dst_folder = os.path.join(dst_root, fr)
                    for filename in os.listdir(src_folder_path):
                        src_path = os.path.join(src_folder_path, filename)
                        if os.path.isfile(src_path):
                            dst_path = os.path.join(dst_root, f"{fr}/{filename}")
                            shutil.copy(src_path, dst_path)
                            count["FoR"][fr] += 1

# ASV2019
def filter_asv2019(dst_root, count):
    meta = [
        os.path.join("/DETECT/Asvpoof2019/LA/LA/ASVspoof2019_LA_cm_protocols", x) for x 
        in ["ASVspoof2019.LA.cm.train.trn.txt", "ASVspoof2019.LA.cm.eval.trl.txt", "ASVspoof2019.LA.cm.dev.trl.txt"]
    ]

    src_root = "/DETECT/Asvpoof2019/LA/LA/"
    dst_root = "/DETECT"
    # /DETECT/Asvpoof2019/LA/LA/ASVspoof2019_LA_dev/flac/LA_D_1000265.flac
    for i, meta_file in enumerate(meta):
        f = open(meta_file, 'r')
        lines = f.readlines()
        for line in lines:
            label = "fake" # Fake 
            if 'bonafide' in line: label = "real" # Real # ex) LA_0069 LA_D_1047731 - - bonafide
            splited = line.split(' ')
            flac_filename = splited[1]
            src_path = os.path.join(src_root, f"ASVspoof2019_LA_{meta_file.split('.')[-3]}/flac/{flac_filename}.flac")
           # print(src_path)
            if os.path.isfile(src_path):
                #print("**")
                dst_path = os.path.join(dst_root, f"{label}/{flac_filename}.flac")
                shutil.copy(src_path, dst_path)
                count["Asv"][label] += 1

if __name__ == '__main__':
    dst_root = "/DETECT"
    os.makedirs(os.path.join(dst_root, "fake"), exist_ok=True)
    os.makedirs(os.path.join(dst_root, "real"), exist_ok=True)
    
    count = {k: {"fake": 0, "real": 0} for k in ["FoR", "Asv"]}
    filter_asv2019(dst_root, count)
    #print(count)
    filter_for(dst_root, count)
    #print(count)
    
    print("Total # of samples:", sum([sum(v1.values()) for v1 in count.values()]))
    print("# of fake samples:", sum([list(v1.values())[0] for v1 in count.values()]))
    print("# of real samples:", sum([list(v1.values())[1] for v1 in count.values()]))
    print(count)
    
'''
LA_0022 LA_E_6977360 - A09 spoof
LA_0031 LA_E_5932896 - A13 spoof
LA_0030 LA_E_5849185 - - bonafide
'''
