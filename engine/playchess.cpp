//#include <mcheck.h>
#include <cstdlib>
#include <cstdio>
#include <list>
#include "chessboard.h"
#include "humanplayer.h"
#include "aiplayer.h"

using namespace std;

int main(int argc, char *argv[]) {

	ChessBoard board;
	list<Move> regulars, nulls;
	int turn = WHITE;
	Move move;
	bool found;



	// Initialize players
	AIPlayer black(BLACK, 3);
	HumanPlayer white(WHITE);

	// setup board
    if(argc < 2) {
	    board.initDefaultSetup();
    }
    else {
        string pos = argv[1];
        string t = argv[2];
        string castle = argv[3];
        string FEN = pos + " " + t + " " + castle;
        board.initFENSetup(FEN);
        if(t.find('b') != std::string::npos)
            turn = BLACK;
    }

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
