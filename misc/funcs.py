# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# KIDs analysis
# misc_funcs.py
# Miscellaneous functions
#
# Marcial Becerril, @ 09 February 2025
# Latest Revision: 09 Feb 2025, 18:45 GMT
#
# For all kind of problems, requests of enhancements and bug reports, please
# write to me at:
#
# mbecerrilt92@gmail.com
# Becerril-TapiaM@cardiff.ac.uk
#
# --------------------------------------------------------------------------------- #


# I N I T I A L   P A R A M E T E R S
# -----------------------------------
HEADER = '\033[95m'     # header
BLUE = '\033[94m'       # blue
GREEN = '\033[92m'      # green
YELLOW = '\033[93m'     # yellow
RED = '\033[91m'        # red
BOLD = '\033[1m'        # bold
UNDERLINE = '\033[4m'   # underline
ENDC = '\033[0m'        # end command


# F U N C T I O N S
# -----------------------------------
def printc(text, alarm):
    """
    Colored message.
    Parameters
    ----------
    text:   [str] message to display.
    alarm:  [str] alarm type to define color.
    ----------
    """

    alarm_colors = [HEADER, BLUE, GREEN, YELLOW, RED, BOLD, UNDERLINE]
    alarm_types = ['title1', 'info', 'ok', 'warn', 'fail', 'title2', 'title3']
    
    try:
        idx_color = alarm_types.index(alarm)
        print(f'{alarm_colors[idx_color]}{text}{ENDC}')

    except ValueError:
        print(text)


def print_tree(data, show_basic_info=False):
    """
    Display all the fields of the file as a tree structure.
    Paramters
    ---------
    data:               [dict] dictionary.
    show_basic_info:    [bool] show item values.
    ---------
    """

    # color and style columns
    column_colors = [BOLD, RED, YELLOW, GREEN, BLUE]
    column_style = ['*', '**', '->', '-', '->']

    # extract all the fields
    fields = explore_fields(data, fields=[])

    # display them
    n = 0
    for field in fields:

        sub_fields = field.split("/")
        tabs = ""

        if n == 0:
            prev_sub = [[]]*len(sub_fields)

        for i, sub in enumerate(sub_fields):

            if prev_sub[i] != sub:

                format_idx = i
        
                if i >= 5:
                    format_idx = 4
        
                msg = tabs + column_colors[format_idx] + column_style[format_idx] + sub + ENDC
        
                if show_basic_info and i == len(sub_fields)-1:

                    info = _get_data_from_tab_line(data, field)
                    print(info)

                    if isinstance(info, bytes) or isinstance(info, int) or isinstance(info, str) or isinstance(info, float):
                        msg += f': {info}'

                    elif isinstance(info, list):
                        msg += f': {len(info)}'

                    else:
                        msg += ": other format"
                
                print(msg)
                
                prev_sub[i+1:] = [[]]*(len(prev_sub)-i+1)

            tabs += "  "

        prev_sub = sub_fields
        n += 1


def explore_fields(data, prev="", fields=[]):
    """ Check dictionary fields. """

    for a in data.keys():

        if isinstance(data[a], dict):

            prev += f'{a}/'
            explore_fields(data[a], fields=fields, prev=prev)
            prev = prev[:-prev[::-1][1:].find('/')-1]

        else:
            fields.append(f'{prev}{a}')

    return fields


def _get_data_from_tab_line(data, addr, sep="/"):
    """ Get data from a defined key dictionary. """

    for i in addr.split(sep):
        try:
            data = data[i]
        except KeyError:
            if i.isnumeric():
                data = data[int(i)]

    return data
