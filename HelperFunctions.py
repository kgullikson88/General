"""
  Just a set of helper functions that I use often

"""
import os


#Ensure a directory exists. Create it if not
def ensure_dir(f):
  d = os.path.dirname(f)
  if not os.path.exists(d):
    os.makedirs(d)
