"""
Transform dump url into download commands.
"""

import subprocess
import json
import os
import sys

archive_url = "https://archive.org/download"


def getarticles(wiki_id):
  dumppath = os.path.join(wiki_id, "dumpstatus.json")
  dumpurl = "{}/{}/dumpstatus.json".format(archive_url, wiki_id)
  os.makedirs(wiki_id)
  subprocess.check_call(["wget", "-q", "-O", dumppath, dumpurl])
  with open(dumppath, "r") as f:
    j = json.loads(f.read())
  files = sorted(j["jobs"]["articlesdump"]["files"].keys())
  for f in files:
    print "{}/{}/{}".format(archive_url, wiki_id, str(f))


def main(argv):
  if len(argv) < 2:
    getarticles("jawiki-20170920")
  else:
    for a in argv[1:]:
      getarticles(a)


if __name__ == "__main__":
  main(sys.argv)
