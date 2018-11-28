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
    std::vector<Action>* act;
    State* s;
    int numState = 1;
    char* res;
    char* resultarray;

    const int threadsPerBlock = 512;

    if(state.is_terminal())
        return Action(state.evaluate_minimax());
    if(!state.white_to_move){
        float value = -INF;
        std::vector<Action> actions;
        state.get_actions(actions);
        const int blocks = (actions.length() + 
                threadsPerBlock - 1)/threadsPerBlock;

        resultarray = (char*)malloc(sizeof(char)*actions.length());
        // execute kernel
        // pass values to GPU
        cudaMalloc((void**)&s, sizeof(State) * numState);
        cudaMalloc((void**)&act, sizeof(Action) * actions.length());
        cudaMalloc((void**)&res, sizeof(char) * actions.length());

        cudaMemcpy(s, state, sizeof(State)*numState, 
                cudaMemcpyHostToDevice);
        cudaMemcpy(act, actions, sizeof(Action)*actions.length(), 
                cudaMemcpyHostToDevice);
        // execute kernel
        minim_kernel<<<blocks, threadsPerBlock>>>(s, act, res);

        cudaThreadSynchronize();

        cudaMemcpy(resultarray, res, sizeof(char)*actions.length(), 
                cudaMemcpyDeviceToHost);

        //free values
        cudaFree(s);
        cudaFree(act);
        cudaFree(res);

        //use resultarray

        //free(resultarray);
        
        for(int i = 0; i < actions.length(); i++) {
            if(resultarray[i] == VICTORY)
                return Action(actions[i]->regular, actions[i]->nulls, value);
            else
                value = max(value, resultarray[i]);
        }
        free(resultarray);

        //value = max(value, minimax(next_state));
        //if(value == VICTORY)
            //return Action(it->regular, it->nulls, value);
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
