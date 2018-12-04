#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "IState.h"
#include <vector>

#define INF 9
#define VICTORY 1
#define LOSS 0
namespace msa{
namespace mcts{

__device__ float
max(float a, float b) {
    return a<b?b:a;
}
__device__ float
min(float a, float b) {
    return a>b?b:a;
}
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
    //if(state.white_to_move){
    else {
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
minim_kernel(State* s, Action *a, char* res, int len) {
    // get State and Action corresponding to index
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= len) return;
    //Assume one state
    State state = State(s->depth, s->white_to_move, s->board);
    state.apply_action(a[index]);
    
    if(state.is_terminal())
        res[index] = state.evaluate_minimax(); 

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
    //std::vector<Action>* act;
    State* s;
    int numState = 1;
    char* res;
    char* resultarray;
    int len;

    const int threadsPerBlock = 512;

    std::vector<Action> actions;
    state.get_actions(actions);
    len = actions.size();
    const int blocks = (len + threadsPerBlock - 1)/threadsPerBlock;
    Action *act = &actions[0];

    if(state.is_terminal())
        return Action(state.evaluate_minimax());
    if(!state.white_to_move){
        float value = -INF;

        resultarray = (char*)malloc(sizeof(char)*actions.size());
        // execute kernel
        // pass values to GPU
        cudaMalloc((void**)&s, sizeof(State) * numState);
        cudaMalloc((void**)&act, sizeof(Action) * len);
        cudaMalloc((void**)&res, sizeof(char) * len);

        cudaMemcpy(s, (void *)&state, sizeof(State)*numState, 
                cudaMemcpyHostToDevice);
        cudaMemcpy(act, actions.data(), sizeof(Action)*len, 
                cudaMemcpyHostToDevice);
        // execute kernel
        minim_kernel<<<blocks, threadsPerBlock>>>(s, act, res, len);

        cudaThreadSynchronize();

        cudaMemcpy(resultarray, res, sizeof(char)*len, 
                cudaMemcpyDeviceToHost);

        //free values
        cudaFree(s);
        cudaFree(act);
        cudaFree(res);

        //use resultarray

        //free(resultarray);
        
        for(int i = 0; i < len; i++) {
            if(resultarray[i] == VICTORY)
                return Action(actions.at(i).regular, actions.at(i).nulls, value);
            else
                value = value < resultarray[i] ? resultarray[i] : value;
        }
        free(resultarray);

        //value = max(value, minimax(next_state));
        //if(value == VICTORY)
            //return Action(it->regular, it->nulls, value);
        return Action(actions.begin()->regular, actions.begin()->nulls, LOSS);
    }
    else {
        float value = INF;
         
        resultarray = (char*)malloc(sizeof(char)*len);
         // execute kernel
         // pass values to GPU
        cudaMalloc((void**)&s, sizeof(State) * numState);
        cudaMalloc((void**)&act, sizeof(Action) * len);
        cudaMalloc((void**)&res, sizeof(char) * len);
         
        cudaMemcpy(s, &state, sizeof(State)*numState,
                cudaMemcpyHostToDevice);
        cudaMemcpy(act, actions.data(), sizeof(Action)*len,
                cudaMemcpyHostToDevice);
         // execute kernel
        minim_kernel<<<blocks, threadsPerBlock>>>(s, act, res, len);
         
        cudaThreadSynchronize();
         
        cudaMemcpy(resultarray, res, sizeof(char)*len,
                cudaMemcpyDeviceToHost);
         
        //free values
        cudaFree(s);
        cudaFree(act);
        cudaFree(res);
        //use resultarray
        
        for(int i = 0; i < len; i++) {
            if(resultarray[i] == LOSS)
              return Action(actions.at(i).regular, actions.at(i).nulls, value);
            else
              value = value > resultarray[i] ? resultarray[i] : value;
        }
        free(resultarray);
        return Action(actions.begin()->regular, actions.begin()->nulls, VICTORY);
  }
}
}
}
