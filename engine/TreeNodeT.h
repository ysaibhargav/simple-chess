/*
   A TreeNode in the decision tree.
   I tried to keep this independent of UCT/MCTS.
   Only contains information / methods related to State, Action, Parent, Children etc. 

*/

#pragma once

#include <memory>
#include <math.h>
#include <vector>
#include <algorithm>
#include <omp.h>

#define NOT_PROVEN -1
#define PROVEN_VICTORY 1
#define PROVEN_LOSS 0 

namespace msa {
  namespace mcts {

    template <class State, typename Action>
      class TreeNodeT {
        typedef std::shared_ptr< TreeNodeT<State, Action> > Ptr;

        public:
        //--------------------------------------------------------------
        TreeNodeT(State& state, TreeNodeT* parent = NULL, bool is_root=false):
          state(state),
          action(),
          parent(parent),
          proven_child(NULL),
          agent_id(!state.agent_id()),
          num_visits(0),
          value(0),
          depth(parent ? parent->depth + 1 : 0),
          proved(NOT_PROVEN),
          is_root(is_root)
          {
          }


        //--------------------------------------------------------------
        // expand by adding a single child
        TreeNodeT* expand() {
          // sanity check that we're not already fully expanded
          if(is_fully_expanded()) return NULL;

          // sanity check that we don't have more children than we do actions
          //assert(children.size() < actions.size()) ;

          // if this is the first expansion and we haven't yet got all of the possible actions
          if(actions.empty()) {
            std::vector< Action > _actions;
            // retrieve list of actions from the state
            state.get_actions(_actions);

            if(!is_root) actions = _actions;
            else {
                int tid = omp_get_thread_num();
                int num_threads = omp_get_num_threads();
                int num_actions = _actions.size();
                int span = (num_actions + num_threads - 1) / num_threads;
                int start_idx = tid * span;
                int end_idx = std::min(tid + span, num_actions);
                actions = std::vector< Action >(_actions.begin() + start_idx, _actions.begin() + end_idx);
            }

            // randomize the order
            std::random_shuffle(actions.begin(), actions.end());
          }

          // add the next action in queue as a child
          return add_child_with_action( actions[children.size()] );
        }


        //--------------------------------------------------------------
        void update(const std::vector<float>& rewards) {
          value += rewards[agent_id];
          num_visits++;
        }


        //--------------------------------------------------------------
        // GETTERS
        // state of the TreeNode
        const State& get_state() const { return state; }

        // the action that led to this state
        const Action& get_action() const { return action; }

        // all children have been expanded and simulated
        bool is_fully_expanded() const { return children.empty() == false && children.size() == actions.size(); }

        // does this TreeNode end the search (i.e. the game)
        bool is_terminal() { return state.is_terminal(); }

        // number of times the TreeNode has been visited
        int get_num_visits() const { return num_visits; }

        // accumulated value (wins)
        float get_value() const { return value; }

        // how deep the TreeNode is in the tree
        int get_depth() const { return depth; }

        // number of children the TreeNode has
        int get_num_children() const { return children.size(); }

        // get the i'th child
        TreeNodeT* get_child(int i) const { return children[i].get(); }

        // get parent
        TreeNodeT* get_parent() const { return parent; }

        //private:
        State state;			// the state of this TreeNode
        Action action;			// the action which led to the state of this TreeNode
        TreeNodeT* parent;		// parent of this TreeNode
        TreeNodeT* proven_child;
        int agent_id;			// agent who made the decision

        int num_visits;			// number of times TreeNode has been visited
        float value;			// value of this TreeNode
        int depth;
        int proved;
        bool is_root;

        std::vector< Ptr > children;	// all current children
        std::vector< Action > actions;			// possible actions from this state


        //--------------------------------------------------------------
        // create a clone of the current state, apply action, and add as child
        TreeNodeT* add_child_with_action(const Action& new_action) {
          State child_state = state;
          child_state.apply_action(new_action);

          // create a new TreeNode with the same state (will get cloned) as this TreeNode
          TreeNodeT* child_node = new TreeNodeT(child_state, this);

          // set the action of the child to be the new action
          child_node->action = new_action;

          // add to children
          children.push_back(Ptr(child_node));

          return child_node;
        }

      };

  }
}
