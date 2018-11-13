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
#inculde "chessplayer.h"

namespace msa {
namespace mcts {

class Action {
    public:
        // TODO(sai): verify constructor
        Action(Move &regular, list<Move> &nulls):
            regular(regular),
            nulls(nulls)
        {
        }

        Move regular;
        list<Move> nulls;
};

class State {

    public:
        State(int depth, int is_white, ChessBoard board):
            depth(depth),
            is_white(is_white),
            board(board)
        {
        }
        // copy and assignment operators should perform a DEEP clone of the given state
        // TODO(sai): verify deep copy for struct members (particularly arrays)
        // will default constructors do?
        //State(const State& other);
        //State& operator = (const State& other);

        // whether or not this state is terminal (reached end)
        bool is_terminal() {
            // for mate in n puzzles, we do not want to look at nodes deeper than n
            if (depth == 0)
                return true;

            int color = get_color(); 
            ChessPlayer::Status status = board.getPlayerStatus(color);
            if (status == ChessPlayer::Checkmate || status == ChessPlayer::Stalemate)
                return true;
            return false;
        }

        //  agent id (zero-based) for agent who is about to make a decision
        int agent_id() {
            return is_white;
        } 

        // apply action to state
        void apply_action(const Action& action) {
            // toggle player color
            is_white = !is_white;
            // if it's white's turn, decrease depth value 
            if (is_white) depth--;
            
            board.move(action.regular);
            for(list<Move>::iterator it=action.nulls.begin(); it!=action.nulls.end(); it++)
                board.move(*it);
        }

        // return possible actions from this state
        void get_actions(std::vector<Action>& actions) {
            // sanity check
            if (is_terminal()) return;            

            list<Move> regulars, nulls;
            // TODO(sai): use shared pointers for nulls
            board.getMoves(get_color(), regulars, regulars, nulls);
            for(list<Move>::iterator it=regulars.begin(); it!=regulars.end(); it++)
                actions.push_back(Action(*it, nulls));
        }

        // get a random action, return false if no actions found
        bool get_random_action(Action& action) {
            if (is_terminal()) return false;

            std::vector<Action>& actions;
            get_actions(actions);
            std::default_random_engine generator;
            std::uniform_int_distribution<int> dis(0, actions.size()-1);
            action = actions[dis(generator)];

            return true;
        }

        // evaluate this state and return a vector of rewards (for each agent)
        const std::vector<float> evaluate() {
            // TODO(sai): implement logic for 2 player AI
            // sanity check
            if (!is_white) return;

            vector<float> victory{0, 1};
            vector<float> loss{1, 0};
            if (depth < 0) return loss;

            ChessPlayer::Status status = board.getPlayerStatus(color);
            if (status == ChessPlayer::Checkmate)
                return victory;            
            
            return loss;
        } 

        // return state as string (for debug purposes)
        std::string to_string() const;

        int get_color() {
            return is_white ? WHITE : BLACK;
        }

        // the root state's depth is initialized with n (mate in n)
        int depth;
        int is_white;
        ChessBoard board;
};

}
}
