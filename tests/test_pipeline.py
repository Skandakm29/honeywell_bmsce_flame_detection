import sys
import unittest
import tempfile
import json
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import torch
from train import detections,preprocess,operating_metrics,Flames
from audit_local import OverlapIndex,validated_boxes
from evaluate import match,size_bucket,false_detection_kind


class PipelineTests(unittest.TestCase):
    def test_letterbox_uses_top_left_bgr_and_raw_values(self):
        image = np.zeros((100,200,3),np.uint8)
        image[:,:,:] = (10,20,30)
        tensor,ratio = preprocess(image,416)
        self.assertAlmostEqual(ratio,2.08)
        self.assertEqual(tuple(tensor.shape),(3,416,416))
        self.assertEqual(tensor[:,0,0].tolist(),[10,20,30])
        self.assertEqual(tensor[:,300,0].tolist(),[114,114,114])

    def test_boxes_restore_and_clip_to_original_image(self):
        prediction = torch.tensor([[208.,104.,500.,300.,.9,.9]])
        boxes,scores = detections(prediction,{'height':100,'width':200})
        self.assertEqual(boxes[0].tolist(),[0,0,200,100])
        self.assertAlmostEqual(scores[0].item(),.81,places=5)

    def test_operating_metrics_penalize_duplicates_and_negative_images(self):
        records=[{'boxes':[[0,0,10,10],[20,20,30,30]]},{'boxes':[]}]
        predictions=[{'image_id':0,'bbox':[0,0,10,10],'score':.9},
                     {'image_id':0,'bbox':[0,0,10,10],'score':.8},
                     {'image_id':1,'bbox':[0,0,10,10],'score':.7}]
        result=operating_metrics(records,predictions,thresholds=[.5])['selected']
        self.assertEqual((result['TP'],result['FP'],result['FN']),(1,2,1))
        self.assertAlmostEqual(result['precision'],1/3)
        self.assertAlmostEqual(result['recall'],.5)
        self.assertAlmostEqual(result['F1'],.4)
        self.assertEqual(result['negative_image_false_positive_rate'],1.0)

    def test_overlap_screen_prioritizes_evaluation_split(self):
        index=OverlapIndex()
        reference={'image':'test.jpg','width':100,'height':100,'sha256':'test',
                   'dhash':123,'thumbnail':bytes([20]*256)}
        index.add(reference,'test')
        exact=dict(reference,image='train.jpg')
        self.assertEqual(index.match(exact,'train')[0],'exact')
        near=dict(reference,image='near.jpg',sha256='near',dhash=122)
        self.assertEqual(index.match(near,'train')[0],'near')
        different=dict(near,thumbnail=bytes([100]*256))
        self.assertIsNone(index.match(different,'train')[0])

    def test_bad_label_quarantine(self):
        with self.assertRaises(ValueError):
            validated_boxes({'width':100,'height':100},[{'bbox':[0,0,200,10]}])

    def test_augmentation_is_repeatable_and_preserves_padding(self):
        import cv2
        with tempfile.TemporaryDirectory() as folder:
            picture=Path(folder)/'fixture.png'
            cv2.imwrite(str(picture),np.full((100,200,3),(10,80,150),dtype=np.uint8))
            manifest=Path(folder)/'manifest.json'
            manifest.write_text(json.dumps([{'image':str(picture),'height':100,'width':200,'boxes':[[20,20,60,80]]}]))
            dataset=Flames(manifest,augment=True)
            first=dataset[0]
            second=dataset[0]
            self.assertTrue(torch.equal(first[0],second[0]))
            self.assertTrue(torch.equal(first[1],second[1]))
            self.assertEqual(first[0][:,300,0].tolist(),[114,114,114])


    def test_failure_matching_agrees_with_operating_metrics(self):
        records=[{'boxes':[[0,0,10,10],[20,20,30,30]]},{'boxes':[]}]
        predictions=[{'image_id':0,'bbox':[0,0,10,10],'score':.9},
                     {'image_id':0,'bbox':[0,0,10,10],'score':.8},
                     {'image_id':1,'bbox':[0,0,10,10],'score':.7}]
        matched,outcomes=match(records[0]['boxes'],predictions[:2],.5)
        self.assertEqual(matched,[True,False])
        self.assertEqual([hit for _,hit in outcomes],[True,False])
        self.assertEqual(false_detection_kind(predictions[1],records[0]['boxes']),'duplicate')
        self.assertEqual(false_detection_kind(predictions[2],records[1]['boxes']),'negative_image')
        self.assertEqual(false_detection_kind({'bbox':[50,50,5,5]},records[0]['boxes']),'background_on_flame_image')
        expected=operating_metrics(records,predictions,thresholds=[.5])['selected']
        self.assertEqual(expected['FN'],matched.count(False))

    def test_size_buckets_cover_all_sizes(self):
        self.assertEqual(size_bucket(0),'<8px')
        self.assertEqual(size_bucket(16),'16-32px')
        self.assertEqual(size_bucket(5000),'>=64px')


if __name__=='__main__':
    unittest.main()
