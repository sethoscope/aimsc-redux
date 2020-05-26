#!/usr/bin/env python3

import logging
import yaml
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, FileType


def main():
    description = ''
    parser = ArgumentParser(description=description,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    fields = ['filename', 'artist', 'album', 'title', 'label', 'role']
    parser.add_argument('-f', '--field', choices=fields, default='filename',
                        help='field to print')
    parser.add_argument('input', nargs='+', type=FileType('r'))
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    for f in args.input:
        songs = yaml.safe_load(f)
        for song in songs:
            print(song[args.field])

if __name__ == '__main__':
    main()
