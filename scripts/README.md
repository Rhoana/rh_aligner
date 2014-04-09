FijiBento
=========


FijiBento breaks down the alignment process in the Fiji library to small modules,
that can be executed using the command line (no need to open Fiji's UI for that).
This folder includes some script to autmate the process of alignment.


Scripts
-------

* multibeam\_import\_tilespec.py - this script takes a multibeam EM data folder (image + pixelCoordinates.txt + metadata.txt files), processes the images, and outputs a single json file in the Tile-Spec format.
* filter\_tiles.py - this script takes a single json file and a bounding box coordinates, and produces a list of single-tile json files that intersect this bounding box.
* create\_sift\_features.py - takes a directory of json files (each with a single tile), iterates over the files and extracts the SIFT features for each file.

