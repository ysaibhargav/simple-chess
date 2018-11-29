/*
   A very simple C++11 Templated MCTS (Monte Carlo Tree Search) implementation with examples for openFrameworks. 

   MCTS Code Based on the Java (Simon Lucas - University of Essex) and Python (Peter Cowling, Ed Powley, Daniel Whitehouse - University of York) impelementations at http://mcts.ai/code/index.html
   */

#pragma once

#include "TreeNodeT.h"
#include "MSALoopTimer.h"
#include "minimax.h"
#include <cfloat>
#include <assert.h>
// Minimax selection criteria constants
#define ALWAYS 0
#define NONZERO_WINS 1

namespace msa {
  namespace mcts {

    // State must comply with State Interface (see IState.h)
    // Action can be anything (which your State class knows how to handle)
    template <class State, typename Action>
      class UCT {
        typedef TreeNodeT<State, Action> TreeNode;

        private:
        LoopTimer timer;
        int iterations;
        bool debug;

        public:
        float uct_k;					// k value in UCT function. default = sqrt(2)
        unsigned int max_iterations;	// do a maximum of this many iterations (0 to run till end)
        unsigned int max_millis;		// run for a maximum of this many milliseconds (0 to run till end)
        unsigned int simulation_depth;	// how many ticks (frames) to run simulation for
        unsigned int minimax_depth_trigger;
        unsigned int minimax_selection_criterion;
        bool use_minimax_rollouts;
        bool use_minimax_selection;

        //--------------------------------------------------------------
        UCT(bool use_minimax_rollouts=false, bool use_minimax_selection=false,
            unsigned int minimax_depth_trigger=-1, unsigned int minimax_selection_criterion=ALWAYS, bool debug=false) :
          iterations(0),
          uct_k( sqrt(2) ), 
          max_iterations( 100000 ),
          max_millis( 0 ),
          simulation_depth( 10 ),
          use_minimax_rollouts(use_minimax_rollouts),
          use_minimax_selection(use_minimax_selection),
          minimax_depth_trigger(minimax_depth_trigger),
          minimax_selection_criterion(minimax_selection_criterion),
          debug(debug)
        {}


        //--------------------------------------------------------------
        const LoopTimer & get_timer() const {
          return timer;
        }

        const int get_iterations() const {
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
        bool run(State& current_state, Action &final_action, unsigned int seed = 1, std::vector<State>* explored_states = nullptr) {
          if (current_state.is_terminal()) return false;

          if (use_minimax_selection && minimax_selection_criterion == ALWAYS) {
            final_action = minimax2(State(current_state));
            return true;
          }

          // initialize timer
          timer.init();

          // initialize root TreeNode with current state
          TreeNode root_node(current_state);
          if(debug) {
            printf("ROOT\n");
            printf("Node color is %d\n", root_node.agent_id);
            printf("Node value is %f\n", root_node.get_value());
            printf("Num visits is %d\n", root_node.get_num_visits());
          }

          TreeNode* best_node = NULL;

          // iterate
          iterations = 0;
          while(true) {
            // indicate start of loop
            timer.loop_start();

            // 1. SELECT. Start at root, dig down into tree using UCT on all fully expanded nodes
            //if(use_minimax_selection && has_child_with_proven_victory(&root_node, *best_node)) { 
            if(use_minimax_selection && root_node.proved != NOT_PROVEN) { 
              best_node = root_node.proven_child;
              if(debug) printf("Found child with proven victory in iteration %d!\n", iterations);
              break;
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
                    (((node->agent_id == BLACK_ID) && (node->get_value() > 0.)) ||
                    ((node->agent_id == WHITE_ID) && (node->get_num_visits() > (int)node->get_value())))){
                assert(minimax_selection_criterion == NONZERO_WINS);
                float black_reward = minimax(node->get_state());
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
              //						assert(node);	// sanity check
            }
            //printf("Selected move is ");
            //node->action.regular.print();
            //node->state.board.print();

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

            // 3. SIMULATE (if not terminal)
            std::vector<float> rewards;
            bool minimax_search_triggered = false;
            if(!found_proven_node && !node->is_terminal()) {
              Action action;
              for(int t = 0; t < simulation_depth; t++) {
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

              // add to history
              if(explored_states) explored_states->push_back(state);
            }

            // 4. BACK PROPAGATION
            if(debug) printf("BACKPROP\n");
            while(node) {
              node->update(rewards);
              //printf("BACKPROP: node value is %d, num visits is %d\n", node->get_value(), node->get_num_visits());
              //printf("BACKPROP: node color is %d, value is %d, num visits is %d\n", node->agent_id, node->get_value(), node->get_num_visits());
              //printf("BACKPROP: value is %d, num visits is %d \n", node->get_value(), node->get_num_visits());
              if(debug) {
                printf("Node color is %d\n", node->agent_id);
                printf("Node value is %f\n", node->get_value());
                printf("Num visits is %d\n", node->get_num_visits());
                node->state.board.print();
              }
              node = node->get_parent();
            }

            // find most visited child
            best_node = get_most_visited_child(&root_node);
            //best_node = get_most_valuable_child(&root_node);

            // indicate end of loop for timer
            timer.loop_end();

            // exit loop if current total run duration (since init) exceeds max_millis
            if(max_millis > 0 && timer.check_duration(max_millis)) break;

            // exit loop if current iterations exceeds max_iterations
            if(max_iterations > 0 && iterations > max_iterations) break;
            iterations++;
          }

          // return best node's action
          if(best_node){
            final_action = best_node->get_action();
            //TreeNode *child = best_node;
            //while(child = get_most_visited_child(child))
            //    child->get_action().regular.print();    
          }

          return true; 
        }


      };
  }
}
