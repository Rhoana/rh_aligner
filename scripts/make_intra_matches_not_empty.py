# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 12:58:59 2016

@author: Raahil
"""

import glob
import json
import random

num_generated_pts = 2

# Look through all the intra matches and find the empty ones
allfiles = glob.glob('intra_54/*.json')
emptyfiles = []
for f in allfiles:
    stuff = json.load(open(f))
    if len(stuff[0]["correspondencePointPairs"]) == 0:
        emptyfiles.append(f)
        
# Read tilespec
tilespec = json.load(open('W04_Sec001.json', 'r'))

for f in allfiles:
    # If it isn't empty, just copy-paste it
    if not(f in emptyfiles):
        newfilestr = 'intra_54_notempty\\' + f[9:]
        json.dump(json.load(open(f, 'r')), open(newfilestr, 'w'), sort_keys=True, indent=4)
        cooldude = json.load(open(f, 'r'))
    else:
        # Open files and get the two tilespecs the images come from
        matches = json.loads(open(f, 'r').read())
        img1, img2 = matches[0]['url1'], matches[0]['url2']
        tile1, tile2 = 0, 0
        for t in tilespec:
            if t['mipmapLevels']['0']['imageUrl'] == img1:
                tile1 = t
            if t['mipmapLevels']['0']['imageUrl'] == img2:
                tile2 = t
        
        # Determine the region of overlap between the two images
        overlapx_min = max(tile1['bbox'][0], tile2['bbox'][0])
        overlapx_max = max(tile1['bbox'][1], tile2['bbox'][1])
        overlapy_min = max(tile1['bbox'][2], tile2['bbox'][2])
        overlapy_max = max(tile1['bbox'][3], tile2['bbox'][3])
        obbox = [overlapx_min, overlapx_max, overlapy_min, overlapy_max]
        xrang, yrang = obbox[1] - obbox[0], obbox[3] - obbox[2]
        if xrang < 0 or yrang < 0:
            print 'They don\'t overlap. Rest of code probably won\'t work'
        
        # Choose four random points in the overlap region - one from each quadrant
        xvals, yvals = [], []
        xvals.append(random.random() * xrang / 2 + obbox[0])
        xvals.append(random.random() * xrang / 2 + obbox[0] + xrang / 2)
        xvals.append(random.random() * xrang / 2 + obbox[0])
        xvals.append(random.random() * xrang / 2 + obbox[0] + xrang / 2)
        
        yvals.append(random.random() * yrang / 2 + obbox[2])
        yvals.append(random.random() * yrang / 2 + obbox[2])
        yvals.append(random.random() * yrang / 2 + obbox[2] + yrang / 2)
        yvals.append(random.random() * yrang / 2 + obbox[2] + yrang / 2)
        
        # Add these four points to the matches dict
        for i in range(0, len(xvals)):
            newpair = {}
            newpair['dist_after_ransac'] = 1.0
            newp1 = {'l': [xvals[i] - tile1['bbox'][0],yvals[i] - tile1['bbox'][2]], 'w': [xvals[i],yvals[i]]}
            newp2 = {'l': [xvals[i] - tile2['bbox'][0],yvals[i] - tile2['bbox'][2]], 'w': [xvals[i],yvals[i]]}
            newpair['p1'] = newp1
            newpair['p2'] = newp2
            matches[0]["correspondencePointPairs"].append(newpair)
        
        # Print out the new matches file
        newfilestr = 'intra_54_notempty\\' + f[9:]
        json.dump(matches, open(newfilestr, 'w'), sort_keys=True, indent=4)