import argparse
import glob
import json
import os
from pathlib import Path
import shutil

from PIL import Image
from tqdm import tqdm

from test import get_args_parser, main

class FGAHOI:
    @classmethod
    def preprocess_generated(cls,
                             generated_path,
                             hico_path):
        """
        only anno_list.json need resize the bbox, according to new generated height and width
        we do not need to process test_hico.json and trainval_hico.json
        
        Args:
            generated_path (str): Path to the directory containing generated images.
            hico_path (str): Path to the real HICO dataset directory.
        """
        test_hico = json.load(open(os.path.join(hico_path, "annotations", "test_hico.json")))
        anno_list = json.load(open(os.path.join(hico_path, "annotations", "anno_list.json")))
        assert len(glob.glob(generated_path+'/*.jpg')) == len(test_hico), \
            f"Mismatched number of generated images {len(glob.glob(generated_path+'/*.jpg'))} != {len(test_hico)}"
        print(f"We have {len(test_hico)} images.")
        
        anno_list_to_idx_map = {x["global_id"]: i for i, x in enumerate(anno_list)}
        
        for anno in tqdm(test_hico, desc="Process"):
            img_id = anno["file_name"].split('.')[0]
            new_img = Image.open(os.path.join(generated_path, f"{img_id}.jpg"))
            t = anno_list[anno_list_to_idx_map[img_id]]
            ori_width = t['image_size'][1]
            ori_height = t['image_size'][0]
            # new_height, new_width = 512, 512
            new_width, new_height = new_img.size
            for hoi in t['hois']:
                for i, bbox in enumerate(hoi['human_bboxes']):
                    bbox = [new_width*bbox[0]/ori_width,
                            new_height*bbox[1]/ori_height,
                            new_width*bbox[2]/ori_width,
                            new_height*bbox[3]/ori_height]
                    bbox = [int(a) for a in bbox]
                    hoi['human_bboxes'][i] = bbox
                for i, bbox in enumerate(hoi['object_bboxes']):
                    bbox = [new_width*bbox[0]/ori_width,
                            new_height*bbox[1]/ori_height,
                            new_width*bbox[2]/ori_width,
                            new_height*bbox[3]/ori_height]
                    bbox = [int(a) for a in bbox]
                    hoi['object_bboxes'][i] = bbox
            t['image_size'] = [new_height, new_width, 3]
        
        os.makedirs(os.path.join(generated_path, "annotations"), exist_ok=True)
        anno_list_path = os.path.join(generated_path, "annotations/anno_list.json")
        json.dump(anno_list, open(anno_list_path, 'w'))
        print(f"Exported corrected annotations to {anno_list_path}")
        test_hico_path = os.path.join(generated_path, "annotations/test_hico.json")
        json.dump(test_hico, open(test_hico_path, 'w'))
        print(f"Copied test_hico.json to {test_hico_path}")
        # copy file
        for file in ["trainval_hico.json", "corre_hico.npy",
                     "hoi_list_new.json", "hoi_id_to_num.json", "file_name_to_obj_cat.json"]:
            shutil.copy(os.path.join(hico_path, "annotations", file), os.path.join(generated_path, "annotations", file))


    @classmethod
    def eval(cls,
             hoi_path,
             output_dir,
             backbone,
             swin_weight_path,
             fgahoi_weight_path,
             img_folder=None):
        
        test_parser = argparse.ArgumentParser(parents=[get_args_parser()])
        test_args = test_parser.parse_args([
            "--output_dir", output_dir,
            "--hoi_path", hoi_path,
            "--img_folder", hoi_path if img_folder is None else img_folder,
            "--backbone", backbone,
            "--pretrained", swin_weight_path,
            "--resume", fgahoi_weight_path,
            "--eval", "--merge", "--hierarchical_merge", "--task_merge",
            "--num_verb_classes", "117", "--num_obj_classes", "80", "--dataset_file", "hico",
        ])
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        return main(test_args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script for FGAHOI")
    parser.add_argument("--hoi_path", required=True, help="Path to the dataset to be evaluated")
    # parser.add_argument("--output_dir", default="logs")
    parser.add_argument("--hicodet_path", default="data/hico_20160224_det")
    parser.add_argument("--backbone", default="swin_tiny", choices=["swin_tiny", "swin_large_384"])
    parser.add_argument("--swin_weight_path", default="param/swin_tiny_patch4_window7_224.pth")
    parser.add_argument("--fgahoi_weight_path", default="weights/FGAHOI_Tiny.pth")
    args = parser.parse_args()
    
    FGAHOI.preprocess_generated(generated_path=args.hoi_path,
                                hico_path=args.hicodet_path)
    
    metric = FGAHOI.eval(
        hoi_path=args.hoi_path,
        # output_dir=args.output_dir,
        output_dir=os.path.join(args.hoi_path, f"eval_results_{args.backbone}"),
        backbone=args.backbone,
        swin_weight_path=args.swin_weight_path,
        fgahoi_weight_path=args.fgahoi_weight_path
    )
    
    print(metric)
    # returns
    # {'mAP_def': np.float64(0.2989469131378525),
    # 'mAP_def_rare': np.float64(0.2241721541234064),
    # 'mAP_def_non_rare': np.float64(0.3212822307655443),
    # 'mAP_ko': np.float64(0.3246692468801841),
    # 'mAP_ko_rare': np.float64(0.2452986165905575),
    # 'mAP_ko_non_rare': np.float64(0.34837735722643615)}
    