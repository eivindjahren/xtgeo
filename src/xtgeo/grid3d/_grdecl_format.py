from contextlib import contextmanager

import xtgeo

xtg = xtgeo.common.XTGeoDialog()

logger = xtg.functionlogger(__name__)


def split_line(line):
    """
    split a keyword line inside a grdecl file. This splits the values of a
    'simple' keyword into tokens. ie.

    >>> list(split_line("3 1.0 3*4 PORO 3*INC 'HELLO WORLD  ' 3*'NAME'"))
    ['3', '1.0', '3*4', 'PORO', '3*INC', "'HELLO WORLD  '", "3*'NAME'"]

    note that we do not require string literals to have delimiting space at the
    end, but at the start. This is to be permissive at the end (as there is no
    formal requirement for spaces at end of string literals), but no space at
    the start of a string literal might indicate a repeating count.

    >>> list(split_line("3'hello world'4"))
    ["3'hello world'", '4']

    """
    value = ""
    inside_str = False
    for char in line:
        if char == "'":
            # Either the start or
            # the end of a string literal
            if inside_str:
                yield value + char
                value = ""
                inside_str = False
            else:
                inside_str = True
                value += char
        elif inside_str:
            # inside a string literal
            value += char
        elif value and value[-1] == "-" and char == "-":
            # a comment
            value = value[0:-1]
            break
        elif char.isspace():
            # delimiting space
            if value:
                yield value
                value = ""
        else:
            value += char
    if value:
        yield value


def until_space(string):
    """
    returns the given string until the first space.
    Similar to string.split(max_split=1)[0] except
    initial spaces are not ignored:
    >>> until_space(" hello")
    ''
    >>> until_space("hello world")
    'hello'

    """
    result = ""
    for w in string:
        if w.isspace():
            return result
        result += w
    return result


def match_keyword(kw1, kw2):
    """
    Perhaps surprisingly, the eclipse input format considers keywords
    as 8 character strings with space denoting end. So PORO, 'PORO ', and
    'PORO    ' are all considered the same keyword.

    >>> match_keyword("PORO", "PORO ")
    True
    >>> match_keyword("PORO", "PERM")
    False

    """
    return until_space(kw1) == until_space(kw2)


def interpret_number(numb):
    """
    Interpret a eclipse token as a number, ie. if it is not
    an integer, try interpreting it as a float. raises
    ValueError if not a number

    >>> interpret_number("3")
    3
    >>> interpret_number("1.0")
    1.0
    >>> interpret_number("1.0E+30")
    1e+30

    """
    try:
        return int(numb)
    except ValueError:
        return float(numb)


def interpret_token(val):
    """
    Interpret a eclipse token, tries to interpret the
    value in the following order:
    * string literal
    * keyword
    * repreated keyword
    * number

    If the token cannot be matched, we default to returning
    the uninterpreted token.

    >>> interpret_token("3")
    [3]
    >>> interpret_token("1.0")
    [1.0]
    >>> interpret_token("'hello'")
    ['hello']
    >>> interpret_token("PORO")
    ['PORO']
    >>> interpret_token("3PORO")
    ['3PORO']
    >>> interpret_token("3PORO")
    ['3PORO']
    >>> interpret_token("3*PORO")
    ['PORO', 'PORO', 'PORO']
    >>> interpret_token("3*'PORO '")
    ['PORO ', 'PORO ', 'PORO ']
    >>> interpret_token("3'PORO '")
    ["3'PORO '"]

    """
    if val[0] == "'" and val[-1] == "'":
        # A string literal
        return [val[1:-1]]
    if val[0].isalpha():
        # A keyword
        return [val]
    if "*" in val:
        multiplicand, value = val.split("*")
        return interpret_token(value) * int(multiplicand)
    try:
        return [interpret_number(val)]
    except ValueError:
        return [val]


IGNORE_ALL = None


@contextmanager
def open_grdecl(grdecl_file, keywords, max_len=None, ignore=IGNORE_ALL):
    """
    Opens the grdecl_file for reading. Is a generator for tuples of keyword /
    values in records of that file. The format of the file must be that of the
    GRID section of a eclipse input DATA file.

    The records looked for must be "simple" ie.  start with the keyword, be
    followed by single word values and ended by a slash ('/').

    .. code-block:: none

        KEYWORD
        value value value /

    reading the above file with :code:`open_grdecl("filename.grdecl",
    keywords="KEYWORD")` will generate :code:`[("KEYWORD", ["value", "value",
    "value"])]`

    open_grdecl does not follow includes, obey skips, parse MESSAGE commands or
    make exception for groups and subrecords.

    Note: trailing spaces of keywords are ignored

    Raises:
        ValueError: when end of file is reached without terminating a keyword,
            or the file contains an unrecognized (or ignored) keyword.

    Args:
        keywords (List[str]): Which keywords to look for, these are expected to
        be at the start of a line in the file  and the respective values
        following on subsequent lines separated by whitespace. Reading of a
        keyword is completed by a final '\'. See example above.

        max_len (int): The maximum significant length of a keyword (Eclipse
        uses 8) ignore (List[str]): Keywords that have no associated data, and
        should be ignored, e.g. ECHO. Defaults to ignore all keywords that are
        not part of the results.


    """

    def read_grdecl(grdecl_stream):
        words = []
        keyword = None

        line_no = 1
        line = grdecl_stream.readline()
        while line:
            if not line:
                continue

            if keyword is None:
                if max_len:
                    snubbed = line[0:max_len]
                else:
                    snubbed = line
                matched_keywords = [kw for kw in keywords if match_keyword(kw, snubbed)]
                if matched_keywords:
                    keyword = matched_keywords[0]
                    logger.debug("Keyword %s found on line %d", keyword, line_no)
                elif ignore is not IGNORE_ALL and snubbed.rsplit() not in ignore:
                    raise ValueError("Unrecognized keyword {line}")

            else:
                for word in split_line(line):
                    if word == "/":
                        yield (keyword, words)
                        keyword = None
                        words = []
                        break
                    words += interpret_token(word)
            line = grdecl_stream.readline()
            line_no += 1

        if keyword is not None:
            raise ValueError(f"Reached end of stream while reading {keyword}")

    with open(grdecl_file, "r") as stream:
        yield read_grdecl(stream)
