/*
   A very simple C++11 Templated MCTS (Monte Carlo Tree Search) implementation with examples for openFrameworks. 

   MCTS Code Based on the Java (Simon Lucas - University of Essex) and Python (Peter Cowling, Ed Powley, Daniel Whitehouse - University of York) impelementations at http://mcts.ai/code/index.html
   */

#pragma once

#include "TreeNodeT.h"
//#include "MSALoopTimer.h"
#include "minimax.h"
#include <cfloat>
#include <assert.h>
#include <omp.h>
#include "mic.h"
#include "chessboard.h"
// Minimax selection criteria constants
#define ALWAYS 0
#define NONZERO_WINS 1
#define BUFSIZE 1024

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<double> dsec;

namespace msa {
  namespace mcts {

    // State must comply with State Interface (see IState.h)
    // Action can be anything (which your State class knows how to handle)
    template <class State, typename Action>
      class UCT {
        typedef TreeNodeT<State, Action> TreeNode;

        private:
        //LoopTimer timer;
        unsigned int iterations;
        bool debug;

        public:
        float uct_k;					// k value in UCT function. default = sqrt(2)
        unsigned int max_iterations;	// do a maximum of this many iterations (0 to run till end)
        unsigned int max_millis;		// run for a maximum of this many milliseconds (0 to run till end)
        unsigned int simulation_depth;	// how many ticks (frames) to run simulation for
        bool use_minimax_rollouts;
        bool use_minimax_selection;
        unsigned int minimax_depth_trigger;
        unsigned int minimax_selection_criterion;
        int num_threads;
        unsigned int seed;

        //--------------------------------------------------------------
        UCT(bool use_minimax_rollouts=false, bool use_minimax_selection=false,
            unsigned int minimax_depth_trigger=-1, unsigned int minimax_selection_criterion=ALWAYS, bool debug=false,
            int num_threads=1, unsigned int seed=unsigned(time(0))) :
          iterations(0),
          debug(debug),
          uct_k( sqrt(2) ), 
          max_iterations( 100000 ),
          max_millis( 0 ),
          simulation_depth( 10 ),
          use_minimax_rollouts(use_minimax_rollouts),
          use_minimax_selection(use_minimax_selection),
          minimax_depth_trigger(minimax_depth_trigger),
          minimax_selection_criterion(minimax_selection_criterion),
          num_threads(num_threads),
          seed(seed)
        {
          std::srand(seed);
        }


        //--------------------------------------------------------------
        //const LoopTimer & get_timer() const {
        //  return timer;
        //}

        int get_iterations() const {
          return iterations;
        }

        //--------------------------------------------------------------
        // get best (immediate) child for given TreeNode based on uct score
        TreeNode* get_best_uct_child(TreeNode* node, float uct_k) const {
          // sanity check
          if(!node->is_fully_expanded()) return NULL;

          float best_utc_score = -std::numeric_limits<float>::max();
          TreeNode* best_node = NULL;

          // iterate all immediate children and find best UTC score
          int num_children = node->get_num_children();
          for(int i = 0; i < num_children; i++) {
            TreeNode* child = node->get_child(i);
            float uct_exploitation = (float)child->get_value() / (child->get_num_visits() + FLT_EPSILON);
            float uct_exploration = sqrt( log((float)node->get_num_visits() + 1) / (child->get_num_visits() + FLT_EPSILON) );
            float uct_score = uct_exploitation + uct_k * uct_exploration;

            if(uct_score > best_utc_score) {
              best_utc_score = uct_score;
              best_node = child;
            }
          }

          return best_node;
        }

        bool has_child_with_proven_victory(TreeNode* root_node, TreeNode &best_child) const {
          bool found = false;
          int num_children = root_node->get_num_children();
          for(int i = 0; i < num_children; i++) {
            TreeNode* child = root_node->get_child(i);
            if(child->proved == VICTORY) {
              found = true;
              best_child = *child; 
              break;
            }
          }
          return found;
        }


        //--------------------------------------------------------------
        TreeNode* get_most_visited_child(TreeNode* node) const {
          int most_visits = -1;
          TreeNode* best_node = NULL;

          // iterate all immediate children and find most visited
          int num_children = node->get_num_children();
          for(int i = 0; i < num_children; i++) {
            TreeNode* child = node->get_child(i);
            if(child->get_num_visits() > most_visits) {
              most_visits = child->get_num_visits();
              best_node = child;
            }
          }

          return best_node;
        }

        //--------------------------------------------------------------
        TreeNode* get_most_valuable_child(TreeNode* node) const {
          int best_value = -1;
          TreeNode* best_node = NULL;

          int num_children = node->get_num_children();
          for(int i = 0; i < num_children; i++) {
            TreeNode* child = node->get_child(i);
            int value = (float)child->get_value() / (child->get_num_visits() + FLT_EPSILON);
            if(value > best_value) {
              best_value = value;
              best_node = child;
            }
          }

          return best_node;
        }


        //--------------------------------------------------------------
        bool run(State& current_state, Action &final_action, int run) {//, std::vector<State>* explored_states = nullptr) {
          if (current_state.is_terminal()) return false;

          if (use_minimax_selection && minimax_selection_criterion == ALWAYS) {
            final_action = minimax2(State(current_state));
            return true;
          }

          // initialize timer
          //timer.init();

          //TreeNode* best_node = NULL;
          Move proven_move;
          bool found_proven_move = false; 

          char square[64];
          for(int pos=0; pos<64; pos++)
            square[pos] = current_state.board.square[pos];
          char black_king_pos = current_state.board.black_king_pos; 
          char white_king_pos = current_state.board.white_king_pos; 
          unsigned int depth = current_state.depth;
          int white_to_move = current_state.white_to_move;

          printf("Beginning offload to Phi\n");
          auto t_start = Clock::now();
          double phi_time = 0;
          #ifdef RUN_MIC /* Use RUN_MIC to distinguish between the target of compilation */

          /* This pragma means we want the code in the following block be executed in 
           ** Xeon Phi.
           **/
          #pragma offload target(mic) \
            inout(square) \
            in(black_king_pos) \
            in(white_king_pos) \
            in(depth) \
            in(white_to_move) \
            in(found_proven_move) \
            nocopy(t_start) \
            inout(phi_time) \
            inout(proven_move)
          #endif
          {
            auto inner_t_start = Clock::now();
            printf("Finished offload to Phi\n");
            ChessBoard _board;
            for(int pos=0; pos<64; pos++)
              _board.square[pos] = square[pos];
            _board.black_king_pos = black_king_pos;
            _board.white_king_pos = white_king_pos;
            
            bool read_found_proven_move;
            State _current_state(depth, white_to_move, _board); 
            // initialize root TreeNode with current state
            TreeNode root_node(_current_state, NULL, true);
            if(debug) {
              printf("ROOT\n");
              printf("Node color is %d\n", root_node.agent_id);
              printf("Node value is %f\n", root_node.get_value());
              printf("Num visits is %d\n", root_node.get_num_visits());
            }

            TreeNode* per_thread_best_nodes[num_threads];

            printf("Starting parallel execution\n");
            // iterate
            omp_set_num_threads(num_threads);
            #pragma omp parallel \
              firstprivate(root_node) \
              shared(proven_move, found_proven_move, per_thread_best_nodes)
            {
              char output_filename[BUFSIZE];
              sprintf(output_filename, "out_%d_%d_%d.txt", num_threads, run, omp_get_thread_num());
              FILE *output_file = fopen(output_filename, "w");

              double atomic_time = 0;
              double selection_time = 0; 
              double minimax_time = 0; 
              double expansion_time = 0;
              double simulation_time = 0;
              double backprop_time = 0;

              #pragma omp for \
                schedule(static)
              for(unsigned int _iterations=0; _iterations<max_iterations; _iterations++) {
                // indicate start of loop
                //timer.loop_start();
                auto atomic_t_start = Clock::now();
                #pragma omp atomic read
                read_found_proven_move = found_proven_move;
                atomic_time += (double)std::chrono::duration_cast<dsec>(Clock::now() - atomic_t_start).count();
                
                if(read_found_proven_move) continue;

                auto selection_t_start = Clock::now();
                // 1. SELECT. Start at root, dig down into tree using UCT on all fully expanded nodes
                if(use_minimax_selection && root_node.proved != NOT_PROVEN) { 
                  TreeNode *best_node = root_node.proven_child; 
                  atomic_t_start = Clock::now();
                  #pragma omp critical
                  {
                    proven_move = Move(best_node->get_action().regular);
                    found_proven_move = true;
                  }
                  atomic_time += (double)std::chrono::duration_cast<dsec>(Clock::now() - atomic_t_start).count();
                  //if(debug)
                  printf("Found child with proven victory in iteration %d by thread %d!\n", _iterations, omp_get_thread_num());
                  continue;
                }

                bool found_proven_node = false;
                TreeNode* node = &root_node;
                while(!node->is_terminal() && node->is_fully_expanded()) {
                  if(use_minimax_selection && node->proved != NOT_PROVEN){
                    found_proven_node = true;
                    break;
                  }
                  node = get_best_uct_child(node, uct_k);

                  if(use_minimax_selection && (node->proved == NOT_PROVEN) &&
                      (node->state.depth <= minimax_depth_trigger) &&
                      (((node->agent_id == BLACK_ID) && (node->get_value() > 0.)) ||
                       ((node->agent_id == WHITE_ID) && (node->get_num_visits() > (int)node->get_value())))){
                    assert(minimax_selection_criterion == NONZERO_WINS);
                    printf("Starting minimax at depth %d from thread %d\n", node->state.depth, omp_get_thread_num());
                    //float black_reward = minimax(node->get_state());
                    auto minimax_t_start = Clock::now();
                    float black_reward = omp_minimax(node->get_state(), found_proven_move);
                    minimax_time += (double)std::chrono::duration_cast<dsec>(Clock::now() - minimax_t_start).count();
                    printf("Minimax from thread %d finished with reward %f\n", omp_get_thread_num(), black_reward);
                    if(black_reward == VICTORY) node->proved = PROVEN_VICTORY;
                    else node->proved = PROVEN_LOSS;
                    found_proven_node = true;
                    if(node->agent_id == BLACK_ID && node->proved == PROVEN_VICTORY && node->parent) {
                      node->parent->proved = PROVEN_VICTORY;
                      node->parent->proven_child = node;
                    }
                    if(node->agent_id == WHITE_ID && node->proved == PROVEN_LOSS && node->parent) {
                      node->parent->proved = PROVEN_LOSS;
                      node->parent->proven_child = node;
                    }
                    break;
                  }
                  if(debug) {
                    printf("Best UCT child's color is %d, value is %f, num visits is %d\n",
                        node->agent_id, node->get_value(), node->get_num_visits());
                    node->action.regular.print();
                    node->state.board.print();
                  }
                }
                selection_time += (double)std::chrono::duration_cast<dsec>(Clock::now() - selection_t_start).count();

                auto expansion_t_start = Clock::now();
                // 2. EXPAND by adding a single child (if not terminal or not fully expanded)
                if(!found_proven_node && !node->is_fully_expanded() && !node->is_terminal()) {
                  node = node->expand();
                  if(debug) {
                    printf("Expanded move is ");
                    node->action.regular.print();
                    node->state.board.print();
                  }
                }

                State state(node->get_state());
                expansion_time += (double)std::chrono::duration_cast<dsec>(Clock::now() - expansion_t_start).count();

                auto simulation_t_start = Clock::now();
                // 3. SIMULATE (if not terminal)
                std::vector<float> rewards;
                bool minimax_search_triggered = false;
                if(!found_proven_node && !node->is_terminal()) {
                  Action action;
                  for(unsigned int t = 0; t < simulation_depth; t++) {
                    if(state.is_terminal()) break;

                    if(use_minimax_rollouts && state.depth <= minimax_depth_trigger){
                      float black_reward = minimax(state);
                      rewards.push_back(black_reward);
                      rewards.push_back(1.-black_reward);
                      minimax_search_triggered = true;
                      break;
                    }
                    state.get_random_action(action);
                    state.apply_action(action);
                    if(debug) {
                      printf("Depth %d, move is ", state.depth);
                      action.regular.print();
                      state.board.print();
                    }
                  }
                }

                if(found_proven_node){
                  if(node->proved == PROVEN_VICTORY) rewards = {1., 0.};
                  else rewards = {0., 1.};
                }
                else if(!minimax_search_triggered){
                  // get rewards vector for all agents
                  rewards = state.evaluate();
                  if(rewards[BLACK_ID] == VICTORY)
                    printf("Found possible winning sequence in iteration %d\n", _iterations);

                  // add to history
                  //if(explored_states) explored_states->push_back(state);
                }
                simulation_time += (double)std::chrono::duration_cast<dsec>(Clock::now() - simulation_t_start).count();

                auto backprop_t_start = Clock::now();
                // 4. BACK PROPAGATION
                if(debug) printf("BACKPROP\n");
                while(node) {
                  node->update(rewards);
                  if(debug) {
                    printf("Node color is %d\n", node->agent_id);
                    printf("Node value is %f\n", node->get_value());
                    printf("Num visits is %d\n", node->get_num_visits());
                    node->state.board.print();
                  }
                  node = node->get_parent();
                }
                backprop_time += (double)std::chrono::duration_cast<dsec>(Clock::now() - backprop_t_start).count();

                // find most visited child
                per_thread_best_nodes[omp_get_thread_num()] = get_most_visited_child(&root_node);

                // indicate end of loop for timer
                //timer.loop_end();

                // exit loop if current total run duration (since init) exceeds max_millis
                //if(max_millis > 0 && timer.check_duration(max_millis)) break;
              } // end for
              
              if(!found_proven_move){
                printf("Proven move not found!\n");
                int most_visits = -1;
                TreeNode* best_node = NULL;
                for(int node_idx=0; node_idx<num_threads; node_idx++){
                  TreeNode* current_node = per_thread_best_nodes[node_idx]; 
                  int num_visits = current_node->get_num_visits();
                  if(num_visits > most_visits){
                    best_node = current_node;
                    most_visits = num_visits;
                  }
                }
                proven_move = best_node->get_action().regular;
              }            

              phi_time = (double)std::chrono::duration_cast<dsec>(Clock::now() - inner_t_start).count();
              double overall_time = (double)std::chrono::duration_cast<dsec>(Clock::now() - t_start).count();
              fprintf(output_file, "Overall time %.2f\n", overall_time);
              fprintf(output_file, "Phi time %.2f\n", phi_time);
              fprintf(output_file, "Offloading time %.2f\n", overall_time-phi_time);
              fprintf(output_file, "Selection time %.2f\n", selection_time);
              fprintf(output_file, "Expansion time %.2f\n", expansion_time);
              fprintf(output_file, "Simulation time %.2f\n", simulation_time);
              fprintf(output_file, "Backprop time %.2f\n", backprop_time);
              fprintf(output_file, "Minimax time %.2f\n", minimax_time);
              fprintf(output_file, "Atomic time %.2f\n", atomic_time);
              fclose(output_file);
            } // end parallel region
          } // end mic
          printf("Finished parallel execution\n");

          // return best node's action
          //if(best_node){
          //  final_action = best_node->get_action();
          //}
          final_action = Action(proven_move);        

          return true; 
        }


      };
  }
}
