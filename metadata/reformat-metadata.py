#!/usr/bin/env python3
# Reads historical metadata format, outputs YAML

import logging
import re
import yaml
import os.path
from collections import defaultdict
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, FileType


def parse_line(line):
    match = re.search('a{(.*?)},'
                      'l{(.*?)},'
                      't{(.*?)},'
                      'f{(.*?)}', line)
    return dict(zip(('artist', 'album', 'title', 'filename'),
                    match.groups()))

def main():
    description = ''
    parser = ArgumentParser(description=description,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('input', nargs='+')
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # hack alert! I require that input filenames be ..../role/label.data
    songs = []
    for filename in args.input:
        role = os.path.basename(os.path.dirname(filename))
        label, _ = os.path.splitext(os.path.basename(filename))
        with open(filename, 'r') as f:
            for line in f:
                song = parse_line(line)
                song['role'] = role
                song['label'] = label
                nope='/mnt/cdrom/'
                if song['filename'].startswith(nope):
                    song['filename'] = song['filename'][len(nope):]
                songs.append(song)
    print(yaml.dump(songs))

if __name__ == '__main__':
    main()
