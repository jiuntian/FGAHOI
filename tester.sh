CUDA_VISIBLE_DEVICES=3 python tester.py --hoi_path ../generated/v13-4k-cfg3-s666

CUDA_VISIBLE_DEVICES=3 python tester.py --hoi_path ../generated/v13-4k-cfg3-s666 --backbone swin_large_384 --swin_weight_path param/swin_large_patch4_window12_384_22k.pth --fgahoi_weight_path weights/FGAHOI_Large.pth