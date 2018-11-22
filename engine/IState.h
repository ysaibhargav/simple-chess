#pragma once

/*
 Interface that the State class needs to comply with.
 It doesn't need to extend this class, in fact, you don't need to include this file
 This is just here for reference.
 'Action' is your custom class containing action information
 See examples for usage
*/

#include <vector>
#include <string>
#include <list>
#include "chessboard.h"
#include "chessplayer.h"

namespace msa {
namespace mcts {

class Action {
    public:
        // TODO(sai): verify constructor
        Action(Move &regular, std::list<Move> &nulls):
            regular(regular),
            nulls(nulls)
        {
        }

        Action() {
            Move move;
            regular = move;
            nulls = std::list<Move>();
        }

        Move regular;
        std::list<Move> nulls;
};

class State {

    public:
        State(int depth, int white_to_move, ChessBoard board):
            depth(depth),
            white_to_move(white_to_move),
            board(board)
        {
        }
        // copy and assignment operators should perform a DEEP clone of the given state
        // TODO(sai): verify deep copy for struct members (particularly arrays)
        // will default constructors do?
        //State(const State& other);
        //State& operator = (const State& other);

        // whether or not this state is terminal (reached end)
        const bool is_terminal() {
            // for mate in n puzzles, we do not want to look at nodes deeper than n
            if (depth == 0 && white_to_move)
                return true;

            int color = get_color(); 
            ChessPlayer::Status status = board.getPlayerStatus(color);
            if (status == ChessPlayer::Checkmate || status == ChessPlayer::Stalemate)
                return true;
            return false;
        }

        //  agent id (zero-based) for agent who is about to make a decision
        const int agent_id() {
            return white_to_move;
        } 

        // apply action to state
        void apply_action(const Action& action) {
            // if it's white's turn, decrease depth value 
            if (!white_to_move) depth--;
            // toggle player color
            white_to_move = !white_to_move;
            
            // execute maintenance moves if any
            for(std::list<Move>::const_iterator it=action.nulls.begin(); it!=action.nulls.end(); it++)
                board.move(*it);
            // execute the regular move
            board.move(action.regular);
        }

        // return possible actions from this state
        void get_actions(std::vector<Action>& actions) {
            // sanity check
            if (is_terminal()) return;            

            std::list<Move> regulars, nulls;
            // TODO(sai): use shared pointers for nulls
            board.getMoves(get_color(), regulars, regulars, nulls);
            for(std::list<Move>::iterator it=regulars.begin(); it!=regulars.end(); it++) {
                if(board.isValidMove(get_color(), *it))
                    actions.push_back(Action(*it, nulls));
            }
        }

        // get a random action, return false if no actions found
        bool get_random_action(Action& action) {
            if (is_terminal()) return false;

            std::vector<Action> actions;
            get_actions(actions);
            if (actions.size() == 0) return false;
            
            // TODO(sai): make the generator a class member
            std::default_random_engine generator;
            std::uniform_int_distribution<int> dis(0, actions.size()-1);
            action = actions[dis(generator)];

            return true;
        }

        // evaluate this state and return a vector of rewards (for each agent)
        const std::vector<float> evaluate() {
            // TODO(sai): implement logic for 2 player AI
            // TODO(sai): sanity check

            // [BLACK, WHITE]
            std::vector<float> victory{1, 0};
            std::vector<float> loss{0, 1};

            // this will only ever be called when it's white to move
            ChessPlayer::Status status = board.getPlayerStatus(get_color());
            // victory only if mate (mate in n puzzles only)
            if (status == ChessPlayer::Checkmate)
                return victory;            
            
            return loss;
        } 

        // return state as string (for debug purposes)
        std::string to_string() const;

        int get_color() {
            return white_to_move ? WHITE : BLACK;
        }

        void get_maintenance_moves(Action &action) {
            std::list<Move> regulars;
            board.getMoves(get_color(), regulars, regulars, action.nulls);
        }

        // the root state's depth is initialized with n (mate in n)
        int depth;
        int white_to_move;
        ChessBoard board;
};

}
}
