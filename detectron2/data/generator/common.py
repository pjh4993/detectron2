from .labelFile import LabelFile
from detectron2.data import MetadataCatalog
import os

class pascalVOCGenerator:
    def __init__(self, cfg):
        super().__init__()
        self.labelFile = LabelFile()
        """
        self.label_names = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        ).get("thing_classes", None)
        """
        self.label_name = ['human_01', 'human_02']

    def genFromPred(self, prediction, filename, imagePath, image_id):
        assert "instances" in prediction

        pred_result = prediction["instances"].get_fields()
        pred_boxes = pred_result["pred_boxes"].tensor.cpu().detach().numpy().tolist()
        #pred_scores = pred_result["scores"]
        pred_classes = pred_result["pred_classes"].cpu().numpy().tolist()
        #pred_level = pred_result["level"]

        #shapes = {label, bndbox}
        shapes = []
        for class_id, bndbox in zip(pred_classes, pred_boxes):
            class_id = int(image_id[1:4])
            shapes.append({
                "label": self.label_name[class_id],
                "bndbox": bndbox,
            })

        self.labelFile.savePascalVocFormat(
            filename=filename,
            shapes=shapes,
            imagePath=imagePath,
            imageData={"height": prediction["instances"].image_size[0], "width": prediction["instances"].image_size[1]},
            )

