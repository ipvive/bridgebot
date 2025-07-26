#!/bin/bash
gsutil cat "$@" | sed -n 's/.*nt|\(.*\)|pg||\s*$/\1/p'
