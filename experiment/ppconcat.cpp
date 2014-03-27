
#include <stdio.h>

// test if we can exploit preprocessor concat of strings
// to create a nice parsing help in-place
//
// result: this does not seem to work with the same macro name

void parse()
{
	#define PARSE_STRING1 "-h\t shows this help\n"
	#define PARSE_STRING2 "-x\t number of samples\n" PARSE_STRING1
	#define PARSE_STRING3 "-x\t number of samples\n" PARSE_STRING2
	#define PARSE_STRING4 "-x\t number of samples\n" PARSE_STRING3
	#define PARSE_STRING_FINAL "-x\t number of samples\n" PARSE_STRING4
	printf("%s", PARSE_STRING_FINAL);
}

int main(void) 
{
	parse();
}

