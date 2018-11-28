#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "IState.h"
#include <vector>

#include "mini.h"

#define INF 9
#define VICTORY 1
#define LOSS 0

__device__ float
minimax(State state){
    if(state.is_terminal())
        return state.evaluate_minimax(); 

    if(!state.white_to_move){
        float value = -INF;
        std::vector<Action> actions;
        state.get_actions(actions); 
        for(std::vector<Action>::iterator it=actions.begin(); 
                it!=actions.end(); it++) {
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
        for(std::vector<Action>::iterator it=actions.begin();
                it!=actions.end(); it++) {
            State next_state = state;
            next_state.apply_action(*it);
            value = min(value, minimax(next_state));
            if(value == LOSS) return value;
        } 
        return VICTORY;
   }
}

__global__ void
minim_kernel(State* s, Action* a, char* res) {
    // get State and Action corresponding to index
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= a.length()) return;
    //Assume one state
    State state = (*s).apply_action(a[index]);
    
    if(state.is_terminal())
        return state.evaluate_minimax(); 

    if(!state.white_to_move){
        float value = -INF;
        std::vector<Action> actions;
        state.get_actions(actions); 
        for(std::vector<Action>::iterator it=actions.begin(); 
                it!=actions.end(); it++) {
            State next_state = state;
            next_state.apply_action(*it);
            value = max(value, minimax(next_state));
            if(value == VICTORY)
                res[index] = value;
        } 
        res[index] = LOSS;
    }

    if(state.white_to_move){
        float value = INF;
        std::vector<Action> actions;
        state.get_actions(actions); 
        for(std::vector<Action>::iterator it=actions.begin(); 
                it!=actions.end(); it++) {
            State next_state = state;
            next_state.apply_action(*it);
            value = min(value, minimax(next_state));
            if(value == LOSS)
                res[index] = value;
        } 
        res[index] = VICTORY;
    }

}

Action
minimaxCuda(State state) {
    std::vector<Action> actions;
    if(state.is_terminal())
        return Action(state.evaluate_minimax());
    if(!state.white_to_move){
        float value = -INF;
        std::vector<Action> actions;
        state.get_actions(actions);
        // execute kernel
        // pass values to GPU

        // execute kernel

        //free values


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
        //execute kernel


        value = min(value, minimax(next_state));
        if(value == LOSS)
            return Action(it->regular, it->nulls, value);
        }
    return Action(actions.begin()->regular, actions.begin()->nulls, VICTORY);
  }
}
