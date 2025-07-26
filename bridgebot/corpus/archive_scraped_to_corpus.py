import hashlib
import html2sentences
import os
import pdb

def listfiles(folder):
    for root, folders, files in os.walk(folder):
        for filename in files:
            yield os.path.join(root, filename)

def last_snapshots(filenames):
    last_filename, last_prefix = None, None
    for fn in filenames:
        prefix = fn[:-(4 + 2 + 2 + 6 + len(".snapshot"))]  # "YYYYMMDDSSSSSS"
        if prefix != last_prefix and last_filename:
            yield last_filename
        last_prefix = prefix
        last_filename = fn
    if last_filename:
        yield last_filename


input_prefix = "data/scrape1"
output_dir = "data/corpus"

for fn in last_snapshots(listfiles(input_prefix)):
    key = bytes(fn.replace(input_prefix + "/", "wayback-machine-scraper/"), "utf8")
    hexkey = hashlib.sha256(key).hexdigest()
    print(key.decode("utf8"), hexkey)
    with open(fn) as fp:
        text = html2sentences.markdown(fp.read())
        ofn = os.path.join(output_dir, hexkey + ".txt")
        with open(ofn, "w") as ofp:
            ofp.write(text)
