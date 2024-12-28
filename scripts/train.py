#!/usr/bin/env python
import sys
# Adding your grok-main directory to the python path
# sys.path.append("/lustre/home/1900017859/grok-main")
import grok
import os

parser = grok.training.add_args()
parser.set_defaults(logdir=os.environ.get("GROK_LOGDIR", "."))
hparams = parser.parse_args()
hparams.datadir = os.path.abspath(hparams.datadir)
hparams.logdir = os.path.abspath(hparams.logdir)

print(hparams)
print(grok.training.train(hparams))
