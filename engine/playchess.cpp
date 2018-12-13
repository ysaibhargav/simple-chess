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
    int depth = 3;
    int white_to_move = 1;
    msa::mcts::State state(depth, white_to_move, board);
    msa::mcts::Action action;

	list<Move> regulars, nulls;
	int turn = WHITE;
	bool found;



	// Initialize players
	bool use_minimax_rollouts = false;
    unsigned minimax_depth_trigger = depth;
    bool use_minimax_selection = true;
    unsigned minimax_selection_criterion = ALWAYS;
    bool debug = false;
	msa::mcts::UCT<msa::mcts::State, msa::mcts::Action> black(use_minimax_rollouts=use_minimax_rollouts,
        use_minimax_selection=use_minimax_selection, minimax_depth_trigger=minimax_depth_trigger,
        minimax_selection_criterion=minimax_selection_criterion, debug=debug);
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

    msa::mcts::Action r = msa::mcts::minimaxCuda(state);

	for(;false;) {
		// show board
		state.board.print();
        if(state.is_terminal())
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

	ChessPlayer::Status status = state.board.getPlayerStatus(WHITE);

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
