/*
 ***************************************************************************************
 *
 * NAME:
 *    grd3d_scan_roffbinary.c
 *
 *
 * DESCRIPTION:
 *    This is a new line of ROFF handling function (from 2018). Here is a
 *    quick scan ROFF binary output and return for example:
 *
 *    NameEntry              ByteposData      LenData     Datatype
 *    scale!xscale           94               1           2 (=float)
 *    zvalues!splitEnz       1122             15990       6 (=byte)
 *
 *    The ByteposData will be to the start of the ACTUAL (numerical) data,
 *    not the keyword/tag start (differs from Eclipse SCAN result here!)
 *
 * ARGUMENTS:
 *    fc              i     Filehandle (stream) to read from
 *    swap            o     SWAP status, 0 of False, 1 if True
 *    tags            o     A long *char where the tags are separated by a |
 *    rectypes        o     An array with record types: 1 = INT, 2 = FLOAT,
 *                          3 = DOUBLE, 4 = CHAR(STRING), 5 = BOOL, 6 = BYTE
 *    reclengths      o     An array with record lengths (no of elements)
 *    recstarts       o     An array with record starts (in bytes)
 *    maxkw           i     Max number of tags possible to read
 *    debug           i     Debug level
 *
 * RETURNS:
 *    Function: Number of keywords read. If problems, a negative value
 *    Resulting vectors will be updated.
 *
 *
 * NOTE:
 *    The ROFF format was developed independent of RMS, so integer varables
 *    in ROFF does not match integer grid parameters in RMS fully.  ROFF
 *    uses a signed int (4 byte). As integer values in RMS are always
 *    unsigned (non-negative) information will be lost if you try to import
 *    negative integer values from ROFF into RMS."
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

typedef enum {
    tag,
    endtag,
    roff_bin,
    tag_name,
    record_type_int,
    record_type_float,
    record_type_bool,
    record_type_byte,
    record_type_char,
    record_type_double,
    record_type_array,
    record_name,
    record_data,
    data_length,
    eof,
    unknown
} token_type;

typedef enum {
    no_error = 0,
    unexpected_token,
    end_of_stream
} token_error;

typedef struct token {
    token_type type;
    long start;
    void * value;
    token_error error;
} token;

void token_free(token * t){
    free(t->value);
}

typedef struct token_list {
    token * tokens;
    size_t length;
    size_t capacity;
} token_list;


token_list * token_list_init(int cap) {
    token_list * result = malloc(sizeof(token_list));
    result->length = 0;
    result->capacity = cap;
    result->tokens = malloc(sizeof(token) * result->capacity);
    return result;
}

void token_list_append(token_list * tl, token * t) {
    if(tl->length >= tl->capacity) {
        tl->capacity += tl->capacity + 1;
        tl->tokens = realloc(tl->tokens, tl->capacity);
    }
    tl->tokens[tl->length++] = *t;
}

void token_list_reset(token_list * tl) {
    for(int i = 0; i < tl->length;i++){
        token_free(&(tl->tokens[i]));
    }
    tl->length = 0;
}

token * token_list_last(token_list * tl) {
    return tl->tokens + tl->length;
}

void token_list_free(token_list * tl) {
    for(int i = 0; i < tl->length;i++){
        token_free(&(tl->tokens[i]));
    }
    free(tl->tokens);
}

bool token_is_error(token t) {
    return t.error != no_error;
}

int take_character_token(FILE * f, long start, token_list * token_buffer) {
    size_t cap = 30;
    char * char_buffer = malloc(sizeof(char) * cap);
    int num_read = 0;
    token result = {
        unknown,
        start,
        char_buffer,
        end_of_stream
    };

    while(fread(char_buffer + num_read, sizeof(char), 1, f) == 1){
        if(char_buffer[num_read] == '\0'){
            result.error = no_error;
            token_list_append(token_buffer, &result);
            return num_read;
        }
        num_read++;
        if(num_read >= cap) {
            cap += cap + 1;
            char_buffer = realloc(char_buffer, sizeof(char) * cap);
        }
    }

    token_list_append(token_buffer, &result);
    return num_read;
}

/**
 * takes the roff-bin token from the start of roff_file and puts
 * it into the token buffer. Returns the number of bytes read.
 * start parameter is the number of characters read from roff_file
 * so far.
 *
 * Any other token results in an error token.
 */
int take_roff_header_token(FILE * roff_file, long start, token_list * token_buffer) {
    char *  char_buffer = malloc(sizeof(char) * 9);
    int num_read = fread(char_buffer, 1, 9, roff_file);
    token result = {
        unknown,
        start,
        char_buffer,
        end_of_stream
    };

    if (num_read != 9) {
        token_list_append(token_buffer, &result);
        char_buffer[num_read] = '\0';
        return num_read;
    }
    if (char_buffer[8] != '\0' || strcmp(char_buffer, "roff-bin")) {
        result.error = unexpected_token;
        token_list_append(token_buffer, &result);
        char_buffer[num_read] = '\0';
        return num_read;
    }

    result.type = roff_bin;
    result.error = no_error;

    token_list_append(token_buffer, &result);
    return num_read;
}

/**
 * takes the tag token from the roff_file and puts
 * it into the token buffer. Returns the number of bytes read.
 * start parameter is the number of characters read from roff_file
 * so far.
 *
 * Any other token results in an error token.
 */
int take_tag_token(FILE * roff_file, long start, token_list * token_buffer) {
    char *  char_buffer = malloc(sizeof(char) * 4);
    int num_read = fread(char_buffer, 1, 4, roff_file);
    token result = {
        unknown,
        start,
        char_buffer,
        end_of_stream
    };

    if (num_read != 4) {
        char_buffer[num_read] = '\0';
        token_list_append(token_buffer, &result);
        return num_read;
    }
    if (char_buffer[3] != '\0' || strcmp(char_buffer, "tag")) {
        result.error = unexpected_token;
        char_buffer[num_read] = '\0';
        token_list_append(token_buffer, &result);
        return num_read;
    }

    result.type = tag;
    result.error = no_error;

    token_list_append(token_buffer, &result);
    return num_read;
}

token_type get_type_token(char * token_str){
    if(!strcmp(token_str, "int")){
        return record_type_int;
    } else if(!strcmp(token_str, "float")){
        return record_type_float;
    } else if(!strcmp(token_str, "array")){
        return record_type_array;
    } else if(!strcmp(token_str, "bool")){
        return record_type_bool;
    } else if(!strcmp(token_str, "byte")){
        return record_type_byte;
    } else if(!strcmp(token_str, "char")){
        return record_type_char;
    } else if(!strcmp(token_str, "double")){
        return record_type_double;
    }
    return unknown;
}

int tokenize_single_record(FILE * roff_file, long start, token_list * token_buffer, size_t size, bool * is_swap) {
    int read = 0;
    char * name = token_list_last(token_buffer)->value;
    void * value = malloc(size);
    token result = {
        record_data,
        start,
        value,
        end_of_stream
    };

    read += fread(&value, size, 1, roff_file);
    if(read != 1){
        result.error = no_error;
    }

    token_list_append(token_buffer, &result);

    if(!strcpy(name, "byteswaptest") && *((int*)result.value)){
        *is_swap = true;
    }
    return size * read;
}


int tokenize_char_record(FILE * roff_file, long start, token_list * token_buffer) {
    int num_read = take_character_token(roff_file, start, token_buffer);
    token_list_last(token_buffer)->type = record_data;
    return num_read;
}

int take_array_len_token(FILE * roff_file, long start, token_list * token_buffer, bool * is_swap){
    int * value = malloc(sizeof(int));
    token array_len_token = {
        data_length,
        start,
        value,
        end_of_stream
    };

    int read = fread(&value, sizeof(int), 1, roff_file);

    if(read != 1){
        token_list_append(token_buffer, &array_len_token);
        return read;
    }

    array_len_token.type = data_length;
    if(*is_swap){
        SWAP_INT(array_len_token.value);
    }
    token_list_append(token_buffer, &array_len_token);

    return read * sizeof(int);
}

int take_array_data(FILE * roff_file, long start, token_list * token_buffer, int length) {
    int array_len = *((int*)token_list_last(token_buffer)->value);

    token array_data_token = {
        record_data,
        start,
        NULL,
        end_of_stream
    };
    if (fseek(roff_file, array_len, SEEK_CUR) != 0){
        return 0;

    }
    return array_len;
}

int tokenize_array_record(FILE * roff_file, long start, token_list * token_buffer, bool * is_swap) {
    int num_read = take_array_len_token(roff_file, start, token_buffer, is_swap);

    if(token_list_last(token_buffer)->error != no_error){
        return num_read;
    }

    return num_read + take_array_data_token(roff_file, start, token_buffer);
}

/**
 * Assumes the last token read was a record type token and tokenizes the
 * remaining record.
 */
int tokenize_record(FILE * roff_file, long start, token_list * token_buffer, bool * is_swap) {
    token_type last_type = token_list_last(token_buffer)->type;

    int num_read = take_character_token(roff_file, start, token_buffer);
    token_list_last(token_buffer)->type = record_name;

    if(token_list_last(token_buffer)->error != no_error){
        return num_read;
    }

    start += num_read;
    switch(last_type) {
        case record_type_array:
            return num_read + tokenize_array_record(roff_file, start, token_buffer, is_swap);
        case record_type_char:
            return num_read + tokenize_char_record(roff_file, start, token_buffer);
        case record_type_int:
            return num_read + tokenize_single_record(roff_file, start, token_buffer, sizeof(int), is_swap);
        case record_type_float:
            return num_read + tokenize_single_record(roff_file, start, token_buffer, sizeof(float), is_swap);
        case record_type_byte:
            return num_read + tokenize_single_record(roff_file, start, token_buffer, sizeof(unsigned char), is_swap);
        case record_type_double:
            return num_read + tokenize_single_record(roff_file, start, token_buffer, sizeof(double), is_swap);
        case record_type_bool:
            return num_read + tokenize_single_record(roff_file, start, token_buffer, sizeof(unsigned char), is_swap);
        default:
            logger_critical(LI, FI, FU, "Unexpected state reached in roff tokenization");
            return 0;
    }
}

int take_endtag_token_or_record(FILE * roff_file, long start, token_list * token_buffer, bool * is_swap) {
    char * char_buffer = malloc(sizeof(char) * 7);
    char * last_char = char_buffer;
    int total_read = 0;
    token result = {
        unknown,
        start,
        char_buffer,
        end_of_stream
    };
    do {
        char * read_char = char_buffer + total_read;
        int num_read = fread(read_char, 1, 1, roff_file);
        if (num_read != 1) {
            result.end = start + total_read;
            char_buffer[total_read] = '\0';
            token_list_append(token_buffer, &result);
            return num_read;
        }
        total_read++;
    } while(total_read < 7 && *last_char != '\0');

    result.error = unexpected_token;

    if(total_read == 7 && *last_char != '\0') {
        result.error = unexpected_token;
        char_buffer[6] = '\0';
        token_list_append(token_buffer, &result);
        return total_read;
    }


    if(!strcmp(char_buffer, "endtag")){
        // is it endtag token?
        result.error = no_error;
        result.type = endtag;
        token_list_append(token_buffer, &result);
    } else {
        // is it record type token?
        token_type record_type = get_type_token(char_buffer);
        if(record_type != unknown){
            result.error = no_error;
            result.type = record_type;
            token_list_append(token_buffer, &result);
            total_read += tokenize_record(roff_file, start+total_read, token_buffer, is_swap);
        }
        // else defaults to unexpected token
    }

    return total_read;
}

int tokenize_tag(FILE * roff_file, long start, token_list * token_buffer) {
    int current = start;
    bool is_swap = false;

    current += take_tag_token(roff_file, current, token_buffer);
    if(token_list_last(token_buffer)->error != no_error){
        return current - start;
    }

    current += take_character_token(roff_file, current, token_buffer);
    if(token_list_last(token_buffer)->error != no_error){
        return current - start;
    }
    token_list_last(token_buffer)->type = tag_name;

    while(token_list_last(token_buffer)->type != endtag){
        current += take_endtag_token_or_record(roff_file, start, token_buffer, &is_swap);
        if(token_list_last(token_buffer)->error != no_error){
            return current - start;
        }
    }

    return current - start;
}



/* ######################################################################### */
/* LOCAL FUNCTIONS                                                           */
/* ######################################################################### */

#define ROFFSTRLEN 200
#define ROFFARRLEN 15
#define TAGRECORDMAX 100
#define TAGDATAMAX 100

int
_roffbinstring(FILE *fc, char *mystring)

{
    /* read a string; return the number of bytes (including 0 termination) */
    int i;
    char mybyte;

    strcpy(mystring, "");

    for (i = 0; i < ROFFSTRLEN; i++) {
        if (fread(&mybyte, 1, 1, fc) == 1) {
            mystring[i] = mybyte;
            if (mybyte == '\0')
                return i + 1;
        } else {
            logger_critical(LI, FI, FU, "Did not reach end of ROFF string");
            return -99;
        }
    }

    return -1;
}

int
_scan_roff_bin_record(FILE *fc,
                      int *swap,
                      char tagname[ROFFSTRLEN],
                      long npos1,
                      long *npos2,
                      int *numrec,
                      char cname[ROFFARRLEN][ROFFSTRLEN],
                      char pname[ROFFARRLEN][ROFFSTRLEN],
                      int cntype[ROFFARRLEN],
                      long bytepos[ROFFARRLEN],
                      long reclen[ROFFARRLEN])
{
    /*
     * tagname: is the name of the tag
     * npos1: is the byte INPUT position in the file
     * npos2: is the byte OUTPUT position, i.e. ready for next tag
     * cname: is the name of the subtag, as "array"
     * cntype: is data type: 1=int, 2=float, 3=double, 4=char, 5=byte
     * rnlen: is the record length, if > 1 then it is an array type.
     *        => if 1, then it may have several sub keys
     */

    /* int swap = 0; */
    int ndat, nrec, i, n, ic;
    int bsize = 0;
    const int FAIL = -88;
    char tmpname[ROFFSTRLEN] = "";
    long ncum = 0;

    char cdum[ROFFSTRLEN] = "";
    int idum;
    float fdum;
    double ddum;
    unsigned char bdum;

    if (fseek(fc, npos1, SEEK_SET) != 0)
        return FAIL;

    ncum = ncum + npos1;

    nrec = 0; /* record counter (subtag) */

    strcpy(tagname, "");

    for (i = 0; i < TAGRECORDMAX; i++) {

        ncum += _roffbinstring(fc, tmpname);

        if (npos1 == 0 && i == 0 && strncmp(tmpname, "roff-bin", 8) != 0) {
            /* not a ROFF binary file! */
            logger_debug(LI, FI, FU, "Not a valid ROFF binary file!");
            return -9;
        }

        if (strncmp(tmpname, "tag", 3) == 0) {
            ncum += _roffbinstring(fc, tagname);

            logger_debug(LI, FI, FU, "Tag name %s", tagname);

            if (strncmp(tagname, "eof", 3) == 0) {
                return 10;
            }

            /* now the rest of the record may contain of multiple e.g.: */
            /* float xoffset   4.61860625E+05 or */
            /* array float data 15990 */
            /* ... until */
            /* endtag */
            for (n = 0; n < TAGDATAMAX; n++) {
                ncum += _roffbinstring(fc, tmpname);

                if (strncmp(tmpname, "endtag", 6) == 0) {
                    *npos2 = ncum;
                    *numrec = nrec;
                    return 0;
                }

                strcpy(pname[nrec], "NAxxx");

                if (strncmp(tmpname, "int", 3) == 0) {
                    ncum += _roffbinstring(fc, cname[nrec]);
                    bytepos[nrec] = ncum;
                    ncum += fread(&idum, sizeof(int), 1, fc) * sizeof(int);

                    /* special treatment of byteswap */
                    if (strncmp(cname[nrec], "byteswaptest", 13) == 0) {
                        if (idum == 1)
                            *swap = 0;
                        if (idum != 1)
                            *swap = 1;
                    }

                    reclen[nrec] = 1;
                    cntype[nrec] = 1;
                    nrec++;
                } else if (strncmp(tmpname, "float", 5) == 0) {
                    ncum += _roffbinstring(fc, cname[nrec]);
                    bytepos[nrec] = ncum;
                    ncum += fread(&fdum, sizeof(float), 1, fc) * sizeof(float);
                    cntype[nrec] = 2;
                    reclen[nrec] = 1;
                    nrec++;

                } else if (strncmp(tmpname, "double", 6) == 0) {
                    /* never in use? */
                    ncum += _roffbinstring(fc, cname[nrec]);
                    bytepos[nrec] = ncum;
                    ncum += fread(&ddum, sizeof(double), 1, fc) * sizeof(double);
                    cntype[nrec] = 3;
                    reclen[nrec] = 1;
                    nrec++;
                } else if (strncmp(tmpname, "char", 4) == 0) {
                    ncum += _roffbinstring(fc, cname[nrec]);
                    bytepos[nrec] = ncum;
                    /* char in ROFF is actually a string: */
                    ncum += _roffbinstring(fc, cdum);
                    cntype[nrec] = 4;
                    reclen[nrec] = 1;

                    /* special treatment of parameter names (extra info) */
                    if (strncmp(cname[nrec], "name", 4) == 0) {
                        if (strnlen(cdum, ROFFSTRLEN) == 0)
                            strcpy(cdum, "unknown");
                        strcpy(pname[nrec], cdum);
                    }
                    nrec++;
                } else if (strncmp(tmpname, "bool", 4) == 0) {
                    ncum += _roffbinstring(fc, cname[nrec]);
                    bytepos[nrec] = ncum;
                    ncum += fread(&bdum, sizeof(unsigned char), 1, fc) *
                            sizeof(unsigned char);
                    cntype[nrec] = 5;
                    reclen[nrec] = 1;
                    nrec++;
                } else if (strncmp(tmpname, "byte", 4) == 0) {
                    ncum += _roffbinstring(fc, cname[nrec]);
                    bytepos[nrec] = ncum;
                    ncum += fread(&bdum, sizeof(unsigned char), 1, fc) *
                            sizeof(unsigned char);
                    cntype[nrec] = 6;
                    reclen[nrec] = 1;
                    nrec++;
                } else if (strncmp(tmpname, "array", 5) == 0) {
                    ncum += _roffbinstring(fc, tmpname);

                    if (strncmp(tmpname, "int", 3) == 0) {
                        bsize = 4;
                        ncum += _roffbinstring(fc, cname[nrec]);
                        ncum += fread(&ndat, sizeof(int), 1, fc) * sizeof(int);
                        if (*swap)
                            SWAP_INT(ndat);
                        cntype[nrec] = 1;
                        bytepos[nrec] = ncum;
                        reclen[nrec] = ndat;
                        nrec++;
                    } else if (strncmp(tmpname, "float", 5) == 0) {
                        bsize = 4;
                        ncum += _roffbinstring(fc, cname[nrec]);
                        ncum += fread(&ndat, sizeof(int), 1, fc) * sizeof(int);
                        if (*swap)
                            SWAP_INT(ndat);
                        bytepos[nrec] = ncum;
                        cntype[nrec] = 2;
                        reclen[nrec] = ndat;
                        nrec++;

                    }

                    /* double never in use? */

                    else if (strncmp(tmpname, "char", 4) == 0) {
                        /* Note: arrays of type char (ie strings) have UNKNOWN */
                        /* lenghts; hence need special processing! -> bsize 0 */
                        bsize = 0;
                        ncum += _roffbinstring(fc, cname[nrec]);
                        ncum += fread(&ndat, sizeof(int), 1, fc) * sizeof(int);
                        if (*swap)
                            SWAP_INT(ndat);
                        cntype[nrec] = 4;
                        bytepos[nrec] = ncum;
                        reclen[nrec] = ndat;
                        nrec++;
                    } else if (strncmp(tmpname, "bool", 4) == 0) {
                        bsize = 1;
                        ncum += _roffbinstring(fc, cname[nrec]);
                        ncum += fread(&ndat, sizeof(int), 1, fc) * sizeof(int);
                        if (*swap)
                            SWAP_INT(ndat);
                        bytepos[nrec] = ncum;
                        cntype[nrec] = 5;
                        reclen[nrec] = ndat;
                        nrec++;
                    } else if (strncmp(tmpname, "byte", 4) == 0) {
                        bsize = 1;
                        ncum += _roffbinstring(fc, cname[nrec]);
                        ncum += fread(&ndat, sizeof(int), 1, fc) * sizeof(int);
                        if (*swap)
                            SWAP_INT(ndat);
                        bytepos[nrec] = ncum;
                        cntype[nrec] = 6;
                        reclen[nrec] = ndat;
                        nrec++;
                    }

                    if (bsize == 0) {
                        for (ic = 0; ic < ndat; ic++) {
                            ncum += _roffbinstring(fc, cname[nrec]);
                        }
                    } else {
                        ncum += bsize * ndat;
                        if (fseek(fc, ncum, SEEK_SET) != 0)
                            return FAIL;
                    }
                }
            }
        }
    }

    return EXIT_SUCCESS;
}

/* ######################################################################### */
/* LIBRARY FUNCTION                                                          */
/* ######################################################################### */

long
grd3d_scan_roffbinary(FILE *fc,
                      int *swap,
                      char *tags,
                      int *rectypes,
                      long *reclengths,
                      long *recstarts,
                      long maxkw)
{

    char tagname[ROFFSTRLEN] = "";
    char cname[ROFFARRLEN][ROFFSTRLEN];
    char pname[ROFFARRLEN][ROFFSTRLEN];
    int i, j, numrec, ios, cntype[ROFFARRLEN];
    long npos1, npos2, bytepos[ROFFARRLEN], reclen[ROFFARRLEN];
    long nrec = 0;

    npos1 = 0;

    ios = 0;

    tags[0] = '\0';

    rewind(fc);

    for (i = 0; i < maxkw; i++) {
        tagname[0] = '\0';
        ios = _scan_roff_bin_record(fc, swap, tagname, npos1, &npos2, &numrec, cname,
                                    pname, cntype, bytepos, reclen);

        if (ios == -9) {
            logger_error(LI, FI, FU, "Not a ROFF binary file. STOP!");
            return ios;
        } else if (ios < 0) {
            return -10;
        }

        if (strcmp(tagname, "eof") == 0 || ios == 10)
            break;

        for (j = 0; j < numrec; j++) {
            strcat(tags, tagname);
            strcat(tags, "!");
            strcat(tags, cname[j]);

            /* add a third item if parameter name */
            if (strncmp(cname[j], "name", 4) == 0 &&
                strncmp(pname[j], "NAxxx", 2) != 0) {

                strcat(tags, "!");
                strcat(tags, pname[j]);
            }
            strcat(tags, "|");
            rectypes[nrec] = cntype[j];
            reclengths[nrec] = reclen[j];
            recstarts[nrec] = bytepos[j];
            nrec++;
        }

        npos1 = npos2;
    }
    return nrec;
}
