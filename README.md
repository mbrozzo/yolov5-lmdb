# YOLOv5 with LMDB support
This fork was created in order to add support for a specific dataset format to YOLOv5.
Said dataset format consists of a collection of LMDB files containing images and labels; each key contains one image and its labels.
Keys can be any string, as long as they are a valid file name. The program may break if keys contain characters not allowed in file paths.  

Datasets in the YOLOv5 format can be converted to this LMDB format using the yolo2lmdb.py script.  

The LMDB support is enabled by setting the ```lmdb``` field to ```true``` in the dataset's yaml.
The ```train``` and ```val``` fields must point to the folder containing the ```keys.json``` of each datasets.
Said fields can also be lists of paths.
