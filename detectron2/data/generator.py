import dicttoxml

class pascalVOCGenerator:
    def __init__(self):
        super().__init__()
    
    def genFromPred(self, prediction, output_path, filename):
        assert "instances" in prediction
        result = dict()
        result['annotation'] = dict({
            "folder": "nlosGTImage",
            "filename": filename,
            "path": "./nlosGTImage/" + filename,
            "source": dict({
                "database": "Unknown"
            }),
            "size": dict({
                "width": prediction["instances"]._image_size[1],
                "height": prediction["instances"]._image_size[0],
                "depth": 3
            }),
            "segmented" : 0,
        })

        pred_result = prediction["instances"].get_fields()
        pred_boxes = pred_result["pred_boxes"]
        pred_scores = pred_result["scores"]
        pred_classes = pred_result["pred_classes"]
        pred_level = pred_result["level"]

        object_list = []
        for i in range(len(prediction)):
            box = pred_boxes[i].tensor.flatten()
            object_list.append(dict({
                "name": "object",
                "pose": "Unspecified",
                "truncated": 0,
                "difficult": 0,
                "bndbox": dict({
                    "xmin":box[0].item(),
                    "ymin":box[1].item(),
                    "xmax":box[2].item(),
                    "ymax":box[3].item(),
                }),
            }))
        
        result["annotation"]["object"] = object_list

        return dicttoxml.dicttoxml(result)

       
