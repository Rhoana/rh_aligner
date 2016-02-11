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
allfiles = glob.glob('intra_25/*.json')
emptyfiles = []
for f in allfiles:
    stuff = json.load(open(f))
    if len(stuff[0]["correspondencePointPairs"]) == 0:
        emptyfiles.append(f)
        
# Read tilespec
tilespec = json.loads(open('W04_Sec027.json', 'r').read())

for f in allfiles:
    if not(f in emptyfiles):
        #newfilestr = 'intra_25_notempty\\' + f[9:]
        #json.dump(json.load(open(f, 'r')), open(newfilestr, 'w'), sort_keys=True, indent=4)
        #del newfilestr
        cooldude = json.load(open(f, 'r'))
    else:
        matches = json.loads(open(f, 'r').read())
        img1, img2 = matches[0]['url1'], matches[0]['url2']
        tile1, tile2 = 0, 0
        for t in tilespec:
            if t['mipmapLevels']['0']['imageUrl'] == img1:
                tile1 = t
            if t['mipmapLevels']['0']['imageUrl'] == img2:
                tile2 = t
        
        overlapx_min = max(tile1['bbox'][0], tile2['bbox'][0])
        overlapx_max = max(tile1['bbox'][1], tile2['bbox'][1])
        overlapy_min = max(tile1['bbox'][2], tile2['bbox'][2])
        overlapy_max = max(tile1['bbox'][3], tile2['bbox'][3])
        obbox = [overlapx_min, overlapx_max, overlapy_min, overlapy_max]
        xrang, yrang = obbox[1] - obbox[0], obbox[3] - obbox[2]
        if xrang < 0 or yrang < 0:
            print 'They don\'t overlap'
            
        xval1 = random.random() * xrang / 2 + obbox[0]
        yval1 = random.random() * yrang + obbox[2]
        xval2 = random.random() * xrang / 2 + obbox[0] + xrang / 2
        yval2 = random.random() * yrang + obbox[2]
        
        newpair = {}
        newpair['dist_after_ransac'] = 1.0
        newp1 = {'l': [xval1 - tile1['bbox'][0],yval1 - tile1['bbox'][2]], 'w': [xval1,yval1]}
        newp2 = {'l': [xval1 - tile2['bbox'][0],yval1 - tile2['bbox'][2]], 'w': [xval1,yval1]}
        newpair['p1'] = newp1
        newpair['p2'] = newp2
        matches[0]["correspondencePointPairs"].append(newpair)
        
        newpair = {}
        newpair['dist_after_ransac'] = 0.0
        newp1 = {'l': [xval2 - tile1['bbox'][0],yval2 - tile1['bbox'][2]], 'w': [xval2,yval2]}
        newp2 = {'l': [xval2 - tile2['bbox'][0],yval2 - tile2['bbox'][2]], 'w': [xval2,yval2]}
        newpair['p1'] = newp1
        newpair['p2'] = newp2
        matches[0]["correspondencePointPairs"].append(newpair)
        
        newfilestr = 'intra_25_notempty\\' + f[9:]
        json.dump(matches, open(newfilestr, 'w'), sort_keys=True, indent=4)