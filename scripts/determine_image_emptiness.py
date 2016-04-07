# -*- coding: utf-8 -*-
"""
Created on Thu Apr 07 16:07:34 2016

@author: Raahil
"""

import sys
import argparse
import cv2
import numpy as np

def determineempty(fname, thresh):
    imgf = cv2.imread(fname,0)
    clahe = cv2.createCLAHE()
    claheimg = clahe.apply(imgf)
    
    gradientimg = np.gradient(np.array(claheimg,dtype='float32'))
    vargrad = np.var(gradientimg)
    return vargrad > thresh
    # if vargrad > thresh, then the image is empty

def main():
    print(sys.argv)
    # Command line parser
    parser = argparse.ArgumentParser(description='Given an image, determine if it is empty or has data based on a threshold.')
    parser.add_argument('imgname', metavar='imgname', type=str,
                        help='the image filename')
    parser.add_argument('-t', '--threshold', type=float,
                        help='Empty/Data Threshold (default: 1800)',
                        default=1800)

    args = parser.parse_args()

    isblurry = determineempty(args.imgname, args.threshold)
    print('Image is empty: ' + str(isblurry))

if __name__ == '__main__':
    main()
