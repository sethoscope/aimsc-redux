#!/usr/bin/env python3
#
# Sort of like xargs but with a destination in mind, and replicating
# directory tree structure
# 
# For example:
#   cat all-filenames | ./act-on-file.py --srcdir /tmp/all-my-music --destdir /tmp/music-for-project '/bin/ln -f "{src}" "{dest}"'
#
# or
#
# find * -type f -print0 | act-on-file.py -0 --destdir ../mono 'lame -v --mp3input -m m "{src}" "{dest}"'



import os
import os.path
import sys
import logging
import subprocess
import shlex
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, FileType



# from https://bytes.com/topic/python/answers/41987-canonical-way-dealing-null-separated-lines
def lines(f, newline='\n', leave_newline=False, read_size=8192):
    """Like the normal file iter but you can set what string indicates newline."""
    output_lineend = ('', newline)[leave_newline]
    partial_line = ''
    while True:
        chars_just_read = f.read(read_size)
        if not chars_just_read:
            break
        lines = (partial_line + chars_just_read).split(newline)
        partial_line = lines.pop()
        for line in lines:
            yield line + output_lineend
        if partial_line:
            yield partial_line

def main():
    description = ''
    parser = ArgumentParser(description=description,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--prune', type=int,
                        help='number of leading directories to omit before adding to destdir')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('-0', '--sep0', action='store_true',
                        help='input is null separated')
    parser.add_argument('--srcdir', default="")
    parser.add_argument('--destdir', default="",
                        help='if specified, create directories in dest space and provide {dest} name for command')
    parser.add_argument('cmd', default='echo "{src}"', nargs='?',
                        help='command to run. {dest} will be available if destdir is specified')
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

        
    for src_filename in lines(sys.stdin, newline=('\n', '\0')[args.sep0]):
        logging.debug('src: {}'.format(src_filename))
        filenames = {'src' : os.path.join(args.srcdir, src_filename)}
        if args.destdir:
            dest_filename = os.path.join(args.destdir,
                                         '/'.join(src_filename.split('/')[args.prune:]))
            logging.debug('dest: {}'.format(dest_filename))
            filenames['dest'] = dest_filename
            try:
                os.makedirs(os.path.dirname(dest_filename))
            except FileExistsError:
                pass  # that's fine
        if not os.path.exists(filenames['src']):
            logging.error('{} not found'.format(filenames['src']))
        subprocess.run(shlex.split(args.cmd.format(**filenames)))


if __name__ == '__main__':
    main()
