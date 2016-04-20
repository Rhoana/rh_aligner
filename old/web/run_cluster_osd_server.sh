#!/bin/bash

srun -p interact --pty --mem 8000 -t 4:00:00 --tunnel 8765:8765 python -m SimpleHTTPServer 8765

