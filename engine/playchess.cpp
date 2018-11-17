//#include <mcheck.h>
#include <cstdlib>
#include <cstdio>
#include <list>
#include "chessboard.h"
#include "humanplayer.h"
#include "ofxMSAmcts.h"
#include "IState.h"

using namespace std;

int main(int argc, char *argv[]) {

	ChessBoard board;
	// setup board
	board.initDefaultSetup();

    // TODO(sai): set depth from PGN
    int depth = 4;
    int white_to_move = 1;
    msa::mcts::State state(depth, white_to_move, board);
    msa::mcts::Action action;

	list<Move> regulars, nulls;
	int turn = WHITE;
	bool found;



	// Initialize players
	msa::mcts::UCT<msa::mcts::State, msa::mcts::Action> black;
	HumanPlayer white(WHITE);

    printf("MATE IN %d PUZZLE\n", depth);
	// setup board
    if(argc < 2) {
	    board.initDefaultSetup();
    }
    else {
        string pos = argv[1];
        string t = argv[2];
        string castle = argv[3];
        string FEN = pos + " " + t + " " + castle;
        state.board.initFENSetup(FEN);
        //if(t.find('b') != std::string::npos)
        //    turn = BLACK;
    }

	for(;;) {
		// show board
		state.board.print();
        if(turn == WHITE && state.depth == 0)
            break;

		// query player's choice
		if(turn) {
			found = black.run(state, action);
        }
		else {
			found = white.getMove(state.board, action.regular);
            if (found)
                state.get_maintenance_moves(action);
        }

		if(!found)
			break;

        state.apply_action(action);
		action.regular.print();

		// opponents turn
		turn = TOGGLE_COLOR(turn);
	}

	ChessPlayer::Status status = state.board.getPlayerStatus(turn);

	switch(status)
	{
		case ChessPlayer::Checkmate:
			printf("Checkmate\n");
			break;
		case ChessPlayer::Stalemate:
			printf("Stalemate\n");
			break;
        case ChessPlayer::Normal:
            printf("Failed to solve puzzle!\n");
            break;
	}
}
