#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "IState.h"
#include <vector>
//#include <thrust/device_vector.h>

#define INF 9
#define VICTORY 1
#define LOSS 0
#define MSIZE 100
namespace msa{
namespace mcts{

__device__ float
max_cuda(float a, float b) {
    return a<b?b:a;
}
__device__ float
min_cuda(float a, float b) {
    return a>b?b:a;
}
__device__ float
minimax_cuda(State state){
    //malloc/allocate Move array of length 100
    Move move[MSIZE];
    int i = 0;
    //pass empty array to is_terminal to be filled
    //pass int i to find length filled
    if(state.is_terminal_cuda(move, true, &i))
        return state.evaluate_minimax_cuda(move, &i); 

    if(!state.white_to_move){
        float value = -INF;
        
        //get actions here from move array or pass move array to get_actions_cuda
        
        //thrust::device_vector<Action> actions;
        //state.get_actions_cuda(actions);
        //iterate over actions array        
        for(int j = 0; j < i; j++) {
            State next_state = state;
            //retrieve move and apply it
            
            if(!state.board.isValidMove_cuda(state.get_color(), move[j])) continue;
            next_state.apply_action_cuda(Action_cuda(move[j]));
            
            value = max_cuda(value, minimax_cuda(next_state));
            if(value == VICTORY) return value;
        } 
        return LOSS;
    }
    //if(state.white_to_move){
    else {
        float value = INF;
        //thrust::device_vector<Action> actions;
        //state.get_actions_cuda(actions); 
        for(int j = 0; j < i; j++) {
            State next_state = state;
            
            if(!state.board.isValidMove_cuda(state.get_color(), move[j])) continue;
            next_state.apply_action_cuda(Action_cuda(move[j]));
            
            value = min_cuda(value, minimax_cuda(next_state));
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
    Action_cuda ain = Action_cuda(a[index].regular, a[index].minimax_value);
    //ain.regular = a[index].regular;
    //ain.minimax_value = a[index].minimax_value;
    state.apply_action_cuda(ain);
    
    Move move[MSIZE];
    int i = 0;    
    if(state.is_terminal_cuda(move, true, &i)) {
        res[index] = state.evaluate_minimax_cuda(move, &i);
        return;
    }        

    if(!state.white_to_move){
        float value = -INF;
        //thrust::device_vector<Action> actions;
        //state.get_actions_cuda(actions); 
        for(int j = 0; j < i; j++) {
            State next_state = state;
            //don't need to get actions, we don't set minimax value in search
            if(!state.board.isValidMove_cuda(state.get_color(), move[i])) continue;
            Action_cuda a = Action_cuda(move[j]);
            next_state.apply_action_cuda(a);
            value = max_cuda(value, minimax_cuda(next_state));
            if(value == VICTORY) {
                res[index] = value;
                return;
            }
        } 
        res[index] = LOSS;
        return;
    }

    if(state.white_to_move){
        float value = INF;
        //thrust::device_vector<Action> actions;
        //state.get_actions_cuda(actions); 
        for(int j = 0; j < i; j++) {
            State next_state = state;
            if(!state.board.isValidMove_cuda(state.get_color(), move[i])) continue;
            next_state.apply_action_cuda(Action_cuda(move[j]));
            value = min_cuda(value, minimax_cuda(next_state));
            if(value == LOSS) {
                res[index] = value;
                return;
            }
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
        
        //remove nulls?
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
