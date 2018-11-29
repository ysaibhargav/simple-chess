//#include <mcheck.h>
#include <cstdlib>
#include <cstdio>
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
}

int main(int argc, const char *argv[]) {

  _argc = argc - 1;
  _argv = argv + 1;
  const char *input_filename = get_option_string("-f", NULL);
  //int num_of_threads = get_option_int("-n", 1);
  int depth = get_option_int("-d", -1);

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

  int white_to_move = 0;
  msa::mcts::State state(depth, white_to_move, board);
  msa::mcts::Action action;

  list<Move> regulars, nulls;
  int turn = white_to_move ? WHITE : BLACK;
  bool found;


  // Initialize players
  bool use_minimax_rollouts = false;
  unsigned minimax_depth_trigger = depth;
  bool use_minimax_selection = true;
  unsigned minimax_selection_criterion = NONZERO_WINS;//ALWAYS;
  bool debug = false;
  msa::mcts::UCT<msa::mcts::State, msa::mcts::Action> black(use_minimax_rollouts=use_minimax_rollouts,
      use_minimax_selection=use_minimax_selection, minimax_depth_trigger=minimax_depth_trigger,
      minimax_selection_criterion=minimax_selection_criterion, debug=debug);
  HumanPlayer white(WHITE);


  for(;;) {
    // show board
    state.board.print();
    if(state.is_terminal())
      break;

    // query player's choice
    if(turn) {
      found = black.run(state, action);
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
  }

  ChessPlayer::Status status = state.board.getPlayerStatus(WHITE);

  switch(status)
  {
    case ChessPlayer::Checkmate:
      printf("Checkmate\n");
      break;
    case ChessPlayer::Stalemate:
      printf("Stalemate\n");
      break;
    case ChessPlayer::Normal:
      printf("Failed to solve puzzle!\n");
      break;
    case ChessPlayer::InCheck:
      printf("Failed to solve puzzle!\n");
      break;
  }
}
