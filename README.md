# YOLOv5 with LMDB support
This fork was created in order to add support for a specific dataset format to YOLOv5.

## LMDB dataset format
Said dataset format consists of a collection of LMDB files containing images and labels; each key contains one image and its labels.
Keys can be any string, as long as they are a valid file name. The program may break if keys contain characters not allowed in file paths.
Labels are JSON strings with the following structure:
```
{
  'source': 'Various Placards',
  'file_name': 'file-name.jpg',
  'shape': [600, 400, 3],
  'boxes': [
    [ 0.6638655462184874,
      0.4476987447698745,
      0.06554621848739496,
      0.0899581589958159,
      7],
    [ 0.7294117647058823,
      0.45188284518828453,
      0.05546218487394958,
      0.08158995815899582,
      12]
  ]
}
```
The ```boxes``` array contains the actual labels and is the only required field.
Labels are slightly different from those in the YOLO format.
Specifically, they represent:
  - the x coordinate of the top left corner of the box, normalized by the width of the image,
  - the y coordinate of the top left corner of the box, normalized by the height of the image,
  - the width of the box, normalized by the width of the image,
  - the height of the box, normalized by the height of the image,
  - the class (optional, an integer).

**NOTE: image segmentation is not supported.**

## Converting a YOLO dataset to LMDB
Datasets in the YOLOv5 format can be converted to this LMDB format using the yolo2lmdb.py script.  

## Enabling LMDB dataset support in YOLOv5
The LMDB support is enabled by setting the ```lmdb``` field to ```true``` in the dataset's yaml.
The ```train``` and ```val``` fields must point to the folder containing the ```keys.json``` of each datasets.
Said fields can also be lists of paths.
