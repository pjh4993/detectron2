#python tools/train_net.py --config-file configs/SKU110/retinanet_R_50_FPN_1x.yaml --num-gpus 1 SOLVER.IMS_PER_BATCH 4 VIS_PERIOD 20
python demo/demo.py --config-file configs/COCO-Detection/retinanet_R_50_FPN_1x_2014.yaml --input datasets/novel/test/test_0.jpg datasets/novel/test/test_1.jpg datasets/novel/test/test_2.jpg datasets/novel/test/test_3.jpg datasets/novel/test/test_4.jpg  datasets/novel/test/test_5.jpg  datasets/novel/test/test_6.jpg  datasets/novel/test/test_7.jpg datasets/novel/train/train_3.jpg datasets/novel/train/train_4.jpg --output demo/output/ --opts MODEL.WEIGHTS output/COCO-Detection/retinanet_R_50_FPN_1x/model_final.pth 