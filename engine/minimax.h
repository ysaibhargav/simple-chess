#include <vector>
#include "IState.h"
#define INF 9 
#define VICTORY 1.
#define LOSS 0.

namespace msa {
  namespace mcts {

    float max(float a, float b){
      return a < b ? b : a;
    }

    float min(float a, float b){
      return a > b ? b : a;
    }

    // TODO(sai): implement undo in state class
    float minimax(State state){
      if(state.is_terminal())
        return state.evaluate_minimax(); 

      if(!state.white_to_move){
        float value = -INF;
        std::vector<Action> actions;
        state.get_actions(actions); 
        for(std::vector<Action>::iterator it=actions.begin(); it!=actions.end(); it++) {
          State next_state = state;
          next_state.apply_action(*it);
          value = max(value, minimax(next_state));
          if(value == VICTORY) return value;
        } 
        return LOSS;
      }

      if(state.white_to_move){
        float value = INF;
        std::vector<Action> actions;
        state.get_actions(actions); 
        for(std::vector<Action>::iterator it=actions.begin(); it!=actions.end(); it++) {
          State next_state = state;
          next_state.apply_action(*it);
          value = min(value, minimax(next_state));
          if(value == LOSS) return value;
        } 
        return VICTORY;
      }
    }


    Action minimax2(State state){
      if(state.is_terminal())
        return Action(state.evaluate_minimax()); 

      if(!state.white_to_move){
        float value = -INF;
        std::vector<Action> actions;
        state.get_actions(actions); 
        for(std::vector<Action>::iterator it=actions.begin(); it!=actions.end(); it++) {
          State next_state = state;
          next_state.apply_action(*it);
          value = max(value, minimax(next_state));
          if(value == VICTORY)
            return Action(it->regular, it->nulls, value);
        } 
        return Action(actions.begin()->regular, actions.begin()->nulls, LOSS);
      }

      if(state.white_to_move){
        float value = INF;
        std::vector<Action> actions;
        state.get_actions(actions); 
        for(std::vector<Action>::iterator it=actions.begin(); it!=actions.end(); it++) {
          State next_state = state;
          next_state.apply_action(*it);
          value = min(value, minimax(next_state));
          if(value == LOSS)
            return Action(it->regular, it->nulls, value);
        } 
        return Action(actions.begin()->regular, actions.begin()->nulls, VICTORY);
      }
    }

  }
}
