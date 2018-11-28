//#include <mcheck.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <list>
#include "chessboard.h"
#include "humanplayer.h"
#include "aiplayer.h"

using namespace std;

#define BUFSIZE 1024

static int _argc;
static const char **_argv;

/* Starter code function, don't touch */
const char *get_option_string(const char *option_name,
			      const char *default_value)
{
  for (int i = _argc - 2; i >= 0; i -= 2)
    if (strcmp(_argv[i], option_name) == 0)
      return _argv[i + 1];
  return default_value;
}

/* Starter code function, do not touch */
int get_option_int(const char *option_name, int default_value)
{
  for (int i = _argc - 2; i >= 0; i -= 2)
    if (strcmp(_argv[i], option_name) == 0)
      return atoi(_argv[i + 1]);
  return default_value;
}

/* Starter code function, do not touch */
float get_option_float(const char *option_name, float default_value)
{
  for (int i = _argc - 2; i >= 0; i -= 2)
    if (strcmp(_argv[i], option_name) == 0)
      return (float)atof(_argv[i + 1]);
  return default_value;
}

/* Starter code function, do not touch */
static void show_help(const char *program_path)
{
    printf("Usage: %s OPTIONS\n", program_path);
    printf("\n");
    printf("OPTIONS:\n");
    printf("\t-f <input_filename> (required)\n");
    printf("\t-n <num_of_threads> (required)\n");
}

int main(int argc, const char *argv[]) {

    _argc = argc - 1;
    _argv = argv + 1;
    const char *input_filename = get_option_string("-f", NULL);
    int num_of_threads = get_option_int("-n", 1);

    int error = 0;

    if (input_filename == NULL) {
        printf("Error: You need to specify -f.\n");
        error = 1;
    }

    if (error) {
        show_help(argv[0]);
        return 1;
    }

	ChessBoard board;
	list<Move> regulars, nulls;
	int turn = WHITE;
	Move move;
	bool found;



	// Initialize players
	AIPlayer black(BLACK, 3);
	//HumanPlayer black(BLACK);
    HumanPlayer white(WHITE);

    FILE *input = fopen(input_filename, "r");

    if (!input) {
        printf("Unable to open file: %s.\n", input_filename);
        return -1;
    }
    char _FEN[BUFSIZE];
    fgets(_FEN, BUFSIZE, input);
    board.initFENSetup(std::string(_FEN));

	for(;;) {
		// show board
		board.print();

		// query player's choice
		if(turn)
			found = black.getMove(board, move);
		else
			found = white.getMove(board, move);

		if(!found)
			break;

		// if player has a move get all moves
		regulars.clear();
		nulls.clear();
		board.getMoves(turn, regulars, regulars, nulls);

		// execute maintenance moves
		for(list<Move>::iterator it = nulls.begin(); it != nulls.end(); ++it)
			board.move(*it);

		// execute move
		board.move(move);
		move.print();

		// opponents turn
		turn = TOGGLE_COLOR(turn);
	}

	ChessPlayer::Status status = board.getPlayerStatus(turn);

	switch(status)
	{
		case ChessPlayer::Checkmate:
			printf("Checkmate\n");
			break;
		case ChessPlayer::Stalemate:
			printf("Stalemate\n");
			break;
	}
}
