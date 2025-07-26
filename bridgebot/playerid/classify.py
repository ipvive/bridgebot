import difflib
import tensorflow as tf
import unidecode
import xml.etree.ElementTree as ET

import pdb

flags = tf.compat.v1.flags


FLAGS = flags.FLAGS


flags.DEFINE_string("lin_files", None, "Comma-separated list of input .lin files.")
flags.DEFINE_string("name_counts_file", None, "Output of count_players")
flags.DEFINE_string("id_graph", None, "id URLs file, in graphml format")


def compare_key(blocks):
  lengths = [b.size for b in blocks]
  unmatched = blocks[-1].a + blocks[-1].b - 2 * sum(lengths)
  return tuple(sorted(lengths, reverse=True) + [-unmatched])


def is_full_match(name, title):
  diff = difflib.SequenceMatcher(a=name, b=title, autojunk=False).get_matching_blocks()
  if len(diff) == 2 and diff[0].size == len(name):
    if diff[0].b > 0 and title[diff[0].b - 1].isalpha():
      return False
    b_end = diff[0].b + diff[0].size
    if b_end != len(title) and title[b_end].isalpha():
      return False
    return True
  else:
    asc_title = unidecode.unidecode(title)
    if asc_title == title:
      return False
    else:
      return is_full_match(name, asc_title)


def find_best_match(name, titles):
  best_title = None
  best_diff = None
  best_key = None
  alternatives = []
  for title in titles:
    diff = difflib.SequenceMatcher(a=name, b=title, autojunk=False).get_matching_blocks()
    key = compare_key(diff)
    if not best_key or best_key[0] == key[0]:
      alternatives.append(title)
    if not best_key or key > best_key:
      if best_key and best_key[0] != key[0]:
        alternatives = [title]
      best_key = key
      best_diff = diff
      best_title = title
  full_match = is_full_match(name, best_title)
  ratio = difflib.SequenceMatcher(a=name, b=best_title, autojunk=False).ratio()
  return full_match, len(alternatives), ratio, alternatives, best_title


def main(_):
  titles = []
  input_files = []
  for input_pattern in FLAGS.id_graph.split(","):
    input_files.extend(tf.io.gfile.glob(input_pattern))

  for input_file in input_files:
    tree = ET.parse(input_file)
    root = tree.getroot()
    titles.extend([node[1].text for node in root[-1] if len(node) == 2])
  
  titles = list(frozenset(titles))

  with tf.io.gfile.GFile(FLAGS.name_counts_file, "r") as reader:
    for line in reader:
      parts = line.rstrip().title().split(' ')
      name = " ".join(parts[1:])
      match = find_best_match(name, titles)
      print("\t".join([parts[0], *map(str, match), name]))


if __name__ == '__main__':
  tf.compat.v1.app.run()
