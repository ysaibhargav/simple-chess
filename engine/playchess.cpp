//#include <mcheck.h>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <cstring>
#include <list>
#include "chessboard.h"
#include "humanplayer.h"
#include "ofxMSAmcts.h"
#include "IState.h"

using namespace std;

#define BUFSIZE 1024

static int _argc;
static const char **_argv;

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<double> dsec;

double median(int n, double x[]) {
    double temp;
    int i, j;
    // the following two loops sort the array x in ascending order
    for(i=0; i<n-1; i++) {
        for(j=i+1; j<n; j++) {
            if(x[j] < x[i]) {
                // swap elements
                temp = x[i];
                x[i] = x[j];
                x[j] = temp;
            }
        }
    }

    if(n%2==0) {
        // if there is an even number of elements, return mean of the two elements in the middle
        return((x[n/2] + x[n/2 - 1]) / 2.0);
    } else {
        // else return the element in the middle
        return x[n/2];
    }
}

/* Starter code function, don't touch */
const char *get_option_string(const char *option_name,
    const char *default_value)
{
  for (int i = _argc - 2; i >= 0; i -= 2)
    if (strcmp(_argv[i], option_name) == 0)
      return _argv[i + 1];
  return default_value;
}

/* Starter code function, do not touch */
int get_option_int(const char *option_name, int default_value)
{
  for (int i = _argc - 2; i >= 0; i -= 2)
    if (strcmp(_argv[i], option_name) == 0)
      return atoi(_argv[i + 1]);
  return default_value;
}

/* Starter code function, do not touch */
float get_option_float(const char *option_name, float default_value)
{
  for (int i = _argc - 2; i >= 0; i -= 2)
    if (strcmp(_argv[i], option_name) == 0)
      return (float)atof(_argv[i + 1]);
  return default_value;
}

/* Starter code function, do not touch */
static void show_help(const char *program_path)
{
  printf("Usage: %s OPTIONS\n", program_path);
  printf("\n");
  printf("OPTIONS:\n");
  printf("\t-f <input_filename> (required)\n");
  printf("\t-n <num_of_threads> (required)\n");
  printf("\t-d <depth> (required)\n");
  printf("\t-x <run_on_phi> (optional)\n");
}

int main(int argc, const char *argv[]) {

  _argc = argc - 1;
  _argv = argv + 1;
  const char *input_filename = get_option_string("-f", NULL);
  int num_threads = get_option_int("-n", 1);
  int depth = get_option_int("-d", -1);
  int run_on_phi = get_option_int("-x", 0);
  int num_runs = get_option_int("-r", 1);
  int seed = get_option_int("-s", (int)time(0));
  unsigned minimax_depth_trigger = get_option_int("-h", INF);
  const char *parallel_scheme_str = get_option_string("-m", "TREE_PARALLEL");
  const char *heuristic = get_option_string("-H", "MULTIPLE_VISITS");
  int max_iterations = get_option_int("-N", 1000000);

  int error = 0;

  if (input_filename == NULL) {
    printf("Error: You need to specify -f.\n");
    error = 1;
  }

  if (depth == -1) {
    printf("Error: You need to specify -d.\n");
    error = 1;
  }

  if (error) {
    show_help(argv[0]);
    return 1;
  }

  ChessBoard board;
  FILE *input = fopen(input_filename, "r");

  if (!input) {
    printf("Unable to open file: %s.\n", input_filename);
    return -1;
  }
  char _FEN[BUFSIZE];
  fgets(_FEN, BUFSIZE, input);
  printf("MATE IN %d PUZZLE\n", depth);
  printf("Loaded FEN: %s\n", _FEN);
  board.initFENSetup(std::string(_FEN));

  int white_to_move = run_on_phi ? 0 : 1;
  msa::mcts::State state(depth, white_to_move, board);
  msa::mcts::State _state = state;
  msa::mcts::Action action;

  list<Move> regulars, nulls;
  int turn = white_to_move ? WHITE : BLACK;
  int _turn = turn;
  bool found;


  // Initialize players
  bool use_minimax_rollouts = false;
  //unsigned minimax_depth_trigger = depth;
  bool use_minimax_selection = true;
  //unsigned minimax_selection_criterion = NONZERO_WINS;//ALWAYS;
  unsigned minimax_selection_criterion = MULTIPLE_VISITS;
  if(strcmp(heuristic, "MULTIPLE_VISITS") == 0)
    minimax_selection_criterion = MULTIPLE_VISITS;
  else if(strcmp(heuristic, "ALWAYS") == 0)
    minimax_selection_criterion = ALWAYS;
  else if(strcmp(heuristic, "NONZERO_WINS") == 0)
    minimax_selection_criterion = NONZERO_WINS;
  unsigned parallel_scheme = TREE_PARALLEL;
  if(strcmp(parallel_scheme_str, "TREE_PARALLEL") == 0)
    parallel_scheme = TREE_PARALLEL;
  else if(strcmp(parallel_scheme_str, "ROOT_PARALLEL") == 0)
    parallel_scheme = ROOT_PARALLEL;
  //unsigned parallel_scheme = ROOT_PARALLEL;
  bool debug = false;
  msa::mcts::UCT<msa::mcts::State, msa::mcts::Action> black(max_iterations=max_iterations, use_minimax_rollouts=use_minimax_rollouts,
      use_minimax_selection=use_minimax_selection, minimax_depth_trigger=minimax_depth_trigger,
      minimax_selection_criterion=minimax_selection_criterion, debug=debug, num_threads=num_threads,
      seed=(unsigned)seed, parallel_scheme=parallel_scheme);
  HumanPlayer white(WHITE);

  double times[num_runs]; 
  for(int run=0; run<num_runs; run++) {
    auto t_start = Clock::now();
    state = _state;
    turn = _turn;
    for(;;) {
      // show board
      state.board.print();
      if(state.is_terminal())
        break;

      // query player's choice
      if(turn) {
        found = black.run(state, action, run);
      }
      else {
        found = white.getMove(state.board, action.regular);
        if (found)
          state.get_maintenance_moves(action);
      }

      if(!found)
        break;

      state.apply_action(action);
      action.regular.print();

      // opponents turn
      turn = TOGGLE_COLOR(turn);
      if(run_on_phi)
        break;
    }
    times[run] = std::chrono::duration_cast<dsec>(Clock::now() - t_start).count();
  }
  printf("Median run time is %.2f\n", median(num_runs, times));

  ChessPlayer::Status status = state.board.getPlayerStatus(WHITE);

  switch(status)
  {
    case ChessPlayer::Checkmate:
      printf("Checkmate\n");
      break;
    /*
    case ChessPlayer::Stalemate:
      printf("Stalemate\n");
      break;
    case ChessPlayer::Normal:
      printf("Failed to solve puzzle!\n");
      break;
    case ChessPlayer::InCheck:
      printf("Failed to solve puzzle!\n");
      break;
    */
    default:
      break;
  }

  return 0;
}
