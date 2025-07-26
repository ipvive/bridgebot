import hashlib
import html2sentences
import os
import sys
import urllib
import re
import xml

input_prefix = "data/text"
output_dir = "data/corpus"
wiki_dump_id = sys.argv[1]

def listfiles(folder):
    for root, folders, files in os.walk(folder):
        for filename in files:
            yield os.path.join(root, filename)

for fn in listfiles(input_prefix):
  with open(fn, 'r') as fp:
    html = ""
    for line in fp:
      if line.startswith("<doc"):
        doc_id = re.search('id="(\d+)"', line)[1]
        title = urllib.parse.quote_plus(re.search('title="([^"]+)"', line)[1])
        key = bytes("wikidump/?dumpid={}&curid={}&title={}".format(
           wiki_dump_id, doc_id, title), "utf8")
        hexkey = hashlib.sha256(key).hexdigest()
      elif line.startswith("</doc"):
        print(key.decode("utf8"), hexkey)
        html = xml.sax.saxutils.unescape(html)
        text = html2sentences.markdown(html)
        ofn = os.path.join(output_dir, hexkey + ".txt")
        with open(ofn, "w") as ofp:
            ofp.write(text)
        html = ""  
      else:
        html += line
