while (<>) {
  $pageid = $1 if /<id>(\d+)<\/id>/;
  if (/\[\[([^|#\]]*)/) {
    $lrl = $1;
    $lrl =~ s/\t/ /g;
    print "$pageid\t$lrl\n" if $lrl ne '';
  }
}
