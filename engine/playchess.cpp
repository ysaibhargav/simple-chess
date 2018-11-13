//#include <mcheck.h>
#include <cstdlib>
#include <cstdio>
#include <list>
#include "chessboard.h"
#include "humanplayer.h"
#include "ofxMSAmcts.h"
#include "IState.h"

using namespace std;

int main(void) {

	ChessBoard board;
	// setup board
	board.initDefaultSetup();

    // TODO(sai): set depth from PGN
    int depth = 2;
    int is_white = 0;
    State state(depth, is_white, board);
    Action action;

	list<Move> regulars, nulls;
	int turn = WHITE;
	Move move;
	bool found;

	// Initialize players
	msa::mcts::UCT<State, Action> black();
	HumanPlayer white(WHITE);

	for(;;) {
		// show board
		state.board.print();

		// query player's choice
		if(turn) {
			action = black.run(state);
            found = action != Action();
        }
		else {
			found = white.getMove(state.board, action.move);
            if (found)
                state.get_maintenance_moves(action);
        }

		if(!found)
			break;

        state.apply_action(action);

		move.print();

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
	}
}
