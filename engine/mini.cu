#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "IState.h"
#include "chessboard.h"
#include "chessplayer.h"
#include <vector>
//#include <thrust/device_vector.h>

#define INF 9
#define VICTORY 1
#define LOSS 0
#define MSIZE 100

// Pieces defined in lower 4 bits
#define EMPTY	0x00	// Empty square
#define PAWN	0x01	// Bauer
#define ROOK	0x02	// Turm
#define KNIGHT  0x03	// Springer
#define BISHOP  0x04	// Laeufer
#define QUEEN   0x05	// Koenigin
#define KING	0x06	// Koenig

// Extract figure's type
#define FIGURE(x) (0x0F & x)

// Attributes reside in upper 4 bits
#define SET_BLACK(x) (x | 0x10)
#define IS_BLACK(x)  (0x10 & x)

#define SET_MOVED(x) (x | 0x20)
#define IS_MOVED(x)  (0x20 & x)

// For pawn en passant candidates
#define SET_PASSANT(x)   (x | 0x40)
#define CLEAR_PASSANT(x) (x & 0xbf)
#define IS_PASSANT(x)    (0x40 & x)

// For pawn promotion
#define SET_PROMOTED(x)   (x | 0x80)
#define IS_PROMOTED(x)    (0x80 & x)
#define CLEAR_PROMOTED(x) (x & 0x7f)

// Constants to compare with the macros
#define WHITE 0x00
#define BLACK 0x10
#define TOGGLE_COLOR(x) (0x10 ^ x)

#define DEBUG
#ifdef DEBUG
#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n",
        cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#else
#define cudaCheckError(ans) ans
#endif

namespace msa{
namespace mcts{

enum Position {
		A1 = 0, B1, C1, D1, E1, F1, G1, H1,
		A2, B2, C2, D2, E2, F2, G2, H2,
		A3, B3, C3, D3, E3, F3, G3, H3,
		A4, B4, C4, D4, E4, F4, G4, H4,
		A5, B5, C5, D5, E5, F5, G5, H5,
		A6, B6, C6, D6, E6, F6, G6, H6,
		A7, B7, C7, D7, E7, F7, G7, H7,
		A8, B8, C8, D8, E8, F8, G8, H8
	};
    
__device__ __inline__ void inc(int *i) {
    if(*i < MSIZE-1) (*i) = (*i)+1;
}
    
__device__ bool isVulnerable_C(ChessBoard *b, int pos, int figure)
{
	int target_pos, target_figure, row, col, i,j, end;

	// Determine row and column
	row = pos / 8;
	col = pos % 8;

	// 1. Look for Rooks, Queens and Kings above
	for(target_pos = pos + 8; target_pos < 64; target_pos += 8)
	{
		if((target_figure = b->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(target_figure) != IS_BLACK(figure))
			{
				if((target_pos - pos) == 8)
				{
					if(FIGURE(target_figure) == KING)
						return true;
				}

				if((FIGURE(target_figure) == ROOK) || (FIGURE(target_figure) == QUEEN))
					return true;
			}

			break;
		}
	}

	// 2. Look for Rooks, Queens and Kings below
	for(target_pos = pos - 8; target_pos >= 0; target_pos -= 8)
	{
		if((target_figure = b->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(target_figure) != IS_BLACK(figure))
			{
				if((pos - target_pos) == 8)
				{
					if(FIGURE(target_figure) == KING)
						return true;
				}

				if((FIGURE(target_figure) == ROOK) || (FIGURE(target_figure) == QUEEN))
					return true;
			}

			break;
		}
	}

	// 3. Look for Rooks, Queens and Kings left
	for(target_pos = pos - 1, end = pos - (pos % 8); target_pos >= end; target_pos--)
	{
		if((target_figure = b->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(figure) != IS_BLACK(target_figure))
			{
				if((pos - target_pos) == 1)
				{
					if(FIGURE(target_figure) == KING)
						return true;
				}

				if((FIGURE(target_figure) == ROOK) || (FIGURE(target_figure) == QUEEN))
					return true;
			}

			break;
		}
	}

	// 4. Look for Rooks, Queens and Kings right
	for(target_pos = pos + 1, end = pos + (8 - pos % 8); target_pos < end; target_pos++)
	{
		if((target_figure = b->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(figure) != IS_BLACK(target_figure))
			{
				if((target_pos - pos) == 1)
				{
					if(FIGURE(target_figure) == KING)
						return true;
				}

				if((FIGURE(target_figure) == ROOK) || (FIGURE(target_figure) == QUEEN))
					return true;
			}

			break;
		}
	}

	// 5. Look for Bishops, Queens, Kings and Pawns north-east
	for(i = row + 1, j = col + 1; (i < 8) && (j < 8); i++, j++)
	{
		target_pos = i * 8 + j;
		if((target_figure = b->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(figure) != IS_BLACK(target_figure))
			{
				if((target_pos - pos) == 9)
				{
					if(FIGURE(target_figure) == KING)
						return true;
					else if(!IS_BLACK(figure) && (FIGURE(target_figure) == PAWN))
						return true;
				}

				if((FIGURE(target_figure) == BISHOP) || (FIGURE(target_figure) == QUEEN))
					return true;
			}

			break;
		}
	}
	
	// 6. Look for Bishops, Queens, Kings and Pawns south-east
	for(i = row - 1, j = col + 1; (i >= 0) && (j < 8); i--, j++)
	{
		target_pos = i * 8 + j;
		if((target_figure = b->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(figure) != IS_BLACK(target_figure))
			{
				if((pos - target_pos) == 7)
				{
					if(FIGURE(target_figure) == KING)
						return true;
					else if(IS_BLACK(figure) && (FIGURE(target_figure) == PAWN))
						return true;
				}

				if((FIGURE(target_figure) == BISHOP) || (FIGURE(target_figure) == QUEEN))
					return true;
			}

			break;
		}
	}

	// 7. Look for Bishops, Queens, Kings and Pawns south-west
	for(i = row - 1, j = col - 1; (i >= 0) && (j >= 0); i--, j--)
	{
		target_pos = i * 8 + j;
		if((target_figure = b->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(figure) != IS_BLACK(target_figure))
			{
				if((pos - target_pos) == 9)
				{
					if(FIGURE(target_figure) == KING)
						return true;
					else if(IS_BLACK(figure) && (FIGURE(target_figure) == PAWN))
						return true;
				}

				if((FIGURE(target_figure) == BISHOP) || (FIGURE(target_figure) == QUEEN))
					return true;
			}

			break;
		}
	}

	// 8. Look for Bishops, Queens, Kings and Pawns north-west
	for(i = row + 1, j = col - 1; (i < 8) && (j >= 0); i++, j--)
	{
		target_pos = i * 8 + j;
		if((target_figure = b->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(figure) != IS_BLACK(target_figure))
			{
				if((target_pos - pos) == 7)
				{
					if(FIGURE(target_figure) == KING)
						return true;
					else if(!IS_BLACK(figure) && (FIGURE(target_figure) == PAWN))
						return true;
				}

				if((FIGURE(target_figure) == BISHOP) || (FIGURE(target_figure) == QUEEN))
					return true;
			}

			break;
		}
	}
	
	// 9. Look for Knights in upper positions
	if(row < 6)
	{
		// right
		if(col < 7)
		{
			target_pos = pos + 17;

			if((target_figure = b->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(figure) != IS_BLACK(target_figure))
				{
					if(FIGURE(target_figure) == KNIGHT) return true;
				}
			}
		}
		
		// left
		if(col > 0)
		{
			target_pos = pos + 15;
		
			if((target_figure = b->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(figure) != IS_BLACK(target_figure))
				{
					if(FIGURE(target_figure) == KNIGHT) return true;
				}
			}		
		}
	}
	
	// 10. Look for Knights in lower positions
	if(row > 1)
	{
		// right
		if(col < 7)
		{
			target_pos = pos - 15;
		
			if((target_figure = b->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(figure) != IS_BLACK(target_figure))
				{
					if(FIGURE(target_figure) == KNIGHT) return true;
				}
			}
		}
		
		// left
		if(col > 0)
		{
			target_pos = pos - 17;
		
			if((target_figure = b->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(figure) != IS_BLACK(target_figure))
				{
					if(FIGURE(target_figure) == KNIGHT) return true;
				}
			}		
		}
	}

	// 11. Look for Knights in right positions
	if(col < 6)
	{
		// up
		if(row < 7)
		{
			target_pos = pos + 10;
		
			if((target_figure = b->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(figure) != IS_BLACK(target_figure))
				{
					if(FIGURE(target_figure) == KNIGHT) return true;
				}
			}
		}
		
		// down
		if(row > 0)
		{
			target_pos = pos - 6;
		
			if((target_figure = b->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(figure) != IS_BLACK(target_figure))
				{
					if(FIGURE(target_figure) == KNIGHT) return true;
				}
			}		
		}
	}

	// 12. Look for Knights in left positions
	if(col > 1)
	{
		// up
		if(row < 7)
		{
			target_pos = pos + 6;
		
			if((target_figure = b->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(figure) != IS_BLACK(target_figure))
				{
					if(FIGURE(target_figure) == KNIGHT) return true;
				}
			}
		}
		
		// down
		if(row > 0)
		{
			target_pos = pos - 10;
		
			if((target_figure = b->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(figure) != IS_BLACK(target_figure))
				{
					if(FIGURE(target_figure) == KNIGHT) return true;
				}
			}		
		}
	}	

	return false;
}

__device__ void getPawnMoves_C(ChessBoard *b, int figure, int pos, 
        Move *moves, 
        Move *captures, int *i)
{
	Move new_move;
	int target_pos, target_figure;

	// If pawn was previously en passant candidate victim, it isn't anymore.
	// This is a null move because it has to be executed no matter what.
	if(IS_PASSANT(figure))
	{
		new_move.figure = CLEAR_PASSANT(figure);
		new_move.from = pos;
		new_move.to = pos;
		new_move.capture = figure;
		//null_moves.push_back(new_move);
		
		figure = CLEAR_PASSANT(figure);
	}

	// Of course, we only have to set this once
	new_move.figure = figure;
	new_move.from = pos;

	// 1. One step ahead
	target_pos = IS_BLACK(figure) ? pos - 8 : pos + 8;
	if((target_pos >= 0) && (target_pos < 64))
	{
		if((target_figure = b->square[target_pos]) == EMPTY)
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves[*i] = new_move;
            inc(i);
			
			// 2. Two steps ahead if unmoved
			if(!IS_MOVED(figure))
			{
				target_pos = IS_BLACK(figure) ? pos - 16 : pos + 16;
				if((target_pos >= 0) && (target_pos < 64))
				{
					if((target_figure = b->square[target_pos]) == EMPTY)
					{
						new_move.to = target_pos;
						new_move.capture = target_figure;

						// set passant attribute and clear it later
						new_move.figure = SET_PASSANT(figure);
						moves[*i] = new_move;
                        inc(i);
						new_move.figure = figure;
					}
				}
			} // END 2.
		}
	} // END 1.
	
	// 3. Forward capture (White left; Black right)
	if(pos % 8 != 0)
	{
		target_pos = IS_BLACK(figure) ? pos - 9 : pos + 7;
		if((target_pos >= 0) && (target_pos < 64))
		{
			if((target_figure = b->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(target_figure) != IS_BLACK(figure))
				{
					new_move.to = target_pos;
					new_move.capture = target_figure;
					captures[*i] = new_move;
                    inc(i);
				}
			}
			else
			{
				// En passant?
				target_figure = b->square[pos - 1];
				if(IS_PASSANT(target_figure))
				{
					if(IS_BLACK(target_figure) != IS_BLACK(figure))
					{
						new_move.to = target_pos;
						new_move.capture = target_figure;
						captures[*i] = new_move;
                        inc(i);
					}				
				}
			}
		}
	}
	
	// 4. Forward capture (White right; Black left)
	if(pos % 8 != 7)
	{
		target_pos = IS_BLACK(figure) ? pos - 7 : pos + 9;
		if((target_pos >= 0) && (target_pos < 64))
		{
			if((target_figure = b->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(target_figure) != IS_BLACK(figure))
				{
					new_move.to = target_pos;
					new_move.capture = target_figure;
					captures[*i] = new_move;
                    inc(i);
				}
			}
			else
			{
				// En passant?
				target_figure = b->square[pos + 1];
				if(IS_PASSANT(target_figure))
				{
					if(IS_BLACK(target_figure) != IS_BLACK(figure))
					{
						new_move.to = target_pos;
						new_move.capture = target_figure;
						captures[*i] = new_move;
                        inc(i);
					}				
				}
			}
		}
	}	
}

__device__ void getRookMoves_C(ChessBoard *b, int figure, int pos, 
        Move *moves, 
        Move *captures, int *i)
{
	Move new_move;
	int target_pos, target_figure, end;

	// Of course, we only have to set this once
	new_move.figure = figure;
	new_move.from = pos;

	// 1. Move up
	for(target_pos = pos + 8; target_pos < 64; target_pos += 8)
	{
		if((target_figure = b->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(figure) != IS_BLACK(target_figure))
			{
				new_move.to = target_pos;
				new_move.capture = target_figure;
				captures[*i] = new_move;
                inc(i);
			}
			
			break;
		}
		else
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves[*i] = new_move;
            inc(i);
		}
	}

	// 2. Move down
	for(target_pos = pos - 8; target_pos >= 0; target_pos -= 8)
	{	
		if((target_figure = b->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(figure) != IS_BLACK(target_figure))
			{
				new_move.to = target_pos;
				new_move.capture = target_figure;
				captures[*i] = new_move;
                inc(i);
			}
			
			break;
		}
		else
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves[*i] = new_move;
            inc(i);
		}
	}

	// 3. Move left
	for(target_pos = pos - 1, end = pos - (pos % 8); target_pos >= end; target_pos--)
	{
		if((target_figure = b->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(figure) != IS_BLACK(target_figure))
			{
				new_move.to = target_pos;
				new_move.capture = target_figure;
				captures[*i] = new_move;
                inc(i);
			}
			
			break;
		}
		else
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves[*i] = new_move;
            inc(i);
		}
	}

	// 4. Move right
	for(target_pos = pos + 1, end = pos + (8 - pos % 8); target_pos < end; target_pos++)
	{
		if((target_figure = b->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(figure) != IS_BLACK(target_figure))
			{
				new_move.to = target_pos;
				new_move.capture = target_figure;
				captures[*i] = new_move;
                inc(i);
			}
			
			break;
		}
		else
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves[*i] = new_move;
            inc(i);
		}
	}
}

__device__ void getKnightMoves_C(ChessBoard *b, int figure, int pos, 
        Move *moves, 
        Move *captures, int *i)
{
	Move new_move;
	int target_pos, target_figure, row, col;

	// Of course, we only have to set this once
	new_move.figure = figure;
	new_move.from = pos;

	// Determine row and column
	row = pos / 8;
	col = pos % 8;

	// 1. Upper positions
	if(row < 6)
	{
		// right
		if(col < 7)
		{
			target_pos = pos + 17;
		
			if((target_figure = b->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(figure) != IS_BLACK(target_figure))
				{
					new_move.capture = target_figure;
					new_move.to = target_pos;
					captures[*i] = new_move;
                    inc(i);
				}
			}
			else
			{
				new_move.capture = target_figure;
				new_move.to = target_pos;
				moves[*i] = new_move;
                inc(i);
			}
		}
		
		// left
		if(col > 0)
		{
			target_pos = pos + 15;
		
			if((target_figure = b->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(figure) != IS_BLACK(target_figure))
				{
					new_move.capture = target_figure;
					new_move.to = target_pos;
					captures[*i] = new_move;
                    inc(i);
				}
			}
			else
			{
				new_move.capture = target_figure;
				new_move.to = target_pos;
				moves[*i] = new_move;
                inc(i);
			}		
		}
	}
	
	// 2. Lower positions
	if(row > 1)
	{
		// right
		if(col < 7)
		{
			target_pos = pos - 15;
		
			if((target_figure = b->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(figure) != IS_BLACK(target_figure))
				{
					new_move.capture = target_figure;
					new_move.to = target_pos;
					captures[*i] = new_move;
                    inc(i);
				}
			}
			else
			{
				new_move.capture = target_figure;
				new_move.to = target_pos;
				moves[*i] = new_move;
                inc(i);
			}
		}
		
		// left
		if(col > 0)
		{
			target_pos = pos - 17;
		
			if((target_figure = b->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(figure) != IS_BLACK(target_figure))
				{
					new_move.capture = target_figure;
					new_move.to = target_pos;
					captures[*i] = new_move;
                    inc(i);
				}
			}
			else
			{
				new_move.capture = target_figure;
				new_move.to = target_pos;
				moves[*i] = new_move;
                inc(i);
			}		
		}
	}

	// 3. Right positions
	if(col < 6)
	{
		// up
		if(row < 7)
		{
			target_pos = pos + 10;
		
			if((target_figure = b->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(figure) != IS_BLACK(target_figure))
				{
					new_move.capture = target_figure;
					new_move.to = target_pos;
					captures[*i] = new_move;
                    inc(i);
				}
			}
			else
			{
				new_move.capture = target_figure;
				new_move.to = target_pos;
				moves[*i] = new_move;
                inc(i);
			}
		}
		
		// down
		if(row > 0)
		{
			target_pos = pos - 6;
		
			if((target_figure = b->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(figure) != IS_BLACK(target_figure))
				{
					new_move.capture = target_figure;
					new_move.to = target_pos;
					captures[*i] = new_move;
                    inc(i);
				}
			}
			else
			{
				new_move.capture = target_figure;
				new_move.to = target_pos;
				moves[*i] = new_move;
                inc(i);
			}		
		}
	}

	// 4. Left positions
	if(col > 1)
	{
		// up
		if(row < 7)
		{
			target_pos = pos + 6;
		
			if((target_figure = b->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(figure) != IS_BLACK(target_figure))
				{
					new_move.capture = target_figure;
					new_move.to = target_pos;
					captures[*i] = new_move;
                    inc(i);
				}
			}
			else
			{
				new_move.capture = target_figure;
				new_move.to = target_pos;
				moves[*i] = new_move;
                inc(i);
			}
		}
		
		// down
		if(row > 0)
		{
			target_pos = pos - 10;
		
			if((target_figure = b->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(figure) != IS_BLACK(target_figure))
				{
					new_move.capture = target_figure;
					new_move.to = target_pos;
					captures[*i] = new_move;
                    inc(i);
				}
			}
			else
			{
				new_move.capture = target_figure;
				new_move.to = target_pos;
				moves[*i] = new_move;
                inc(i);
			}		
		}
	}	
}

__device__ void getBishopMoves_C(ChessBoard *b, int figure, int pos, 
        Move *moves, 
        Move *captures, int *loc)
{
	Move new_move;
	int target_pos, target_figure, row, col, i, j;

	// Of course, we only have to set this once
	new_move.figure = figure;
	new_move.from = pos;

	// Determine row and column
	row = pos / 8;
	col = pos % 8;

	// 1. Go north-east
	for(i = row + 1, j = col + 1; (i < 8) && (j < 8); i++, j++)
	{
		target_pos = i * 8 + j;
		if((target_figure = b->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(figure) != IS_BLACK(target_figure))
			{
				new_move.to = target_pos;
				new_move.capture = target_figure;
				captures[*loc] = new_move;
                inc(loc);
			}

			break;
		}
		else
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves[*loc] = new_move;
            inc(loc);
		}
	}
	
	// 2. Go south-east
	for(i = row - 1, j = col + 1; (i >= 0) && (j < 8); i--, j++)
	{
		target_pos = i * 8 + j;
		if((target_figure = b->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(figure) != IS_BLACK(target_figure))
			{
				new_move.to = target_pos;
				new_move.capture = target_figure;
				captures[*loc] = new_move;
                inc(loc);
			}
			
			break;
		}
		else
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves[*loc] = new_move;
            inc(loc);
		}
	}

	// 3. Go south-west
	for(i = row - 1, j = col - 1; (i >= 0) && (j >= 0); i--, j--)
	{
		target_pos = i * 8 + j;
		if((target_figure = b->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(figure) != IS_BLACK(target_figure))
			{
				new_move.to = target_pos;
				new_move.capture = target_figure;
				captures[*loc] = new_move;
                inc(loc);
			}
			
			break;
		}
		else
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves[*loc] = new_move;
            inc(loc);
		}
	}

	// 4. Go north-west
	for(i = row + 1, j = col - 1; (i < 8) && (j >= 0); i++, j--)
	{
		target_pos = i * 8 + j;
		if((target_figure = b->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(figure) != IS_BLACK(target_figure))
			{
				new_move.to = target_pos;
				new_move.capture = target_figure;
				captures[*loc] = new_move;
                inc(loc);
			}
			
			break;
		}
		else
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves[*loc] = new_move;
            inc(loc);
		}
	}
}

__device__ void getQueenMoves_C(ChessBoard *b, int figure, int pos, Move *moves, 
        Move *captures, int *i)
{
	// Queen is just the "cartesian product" of Rook and Bishop
	getRookMoves_C(b, figure, pos, moves, captures, i);
	getBishopMoves_C(b, figure, pos, moves, captures, i);
}


__device__ void getKingMoves_C(ChessBoard *b, int figure, int pos, Move *moves,
        Move *captures, int *i)
{
	Move new_move;
	int target_pos, target_figure, row, col;

	// Of course, we only have to set this once
	new_move.figure = figure;
	new_move.from = pos;

	// Determine row and column
	row = pos / 8;
	col = pos % 8;

	// 1. Move left
	if(col > 0)
	{
		// 1.1 up
		if(row < 7)
		{
			target_pos = pos + 7;
			if((target_figure = b->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(target_figure) != IS_BLACK(figure))
				{
					new_move.capture = target_figure;
					new_move.to = target_pos;
					captures[*i] = new_move;
                    inc(i);
				}
			}
			else
			{
				new_move.to = target_pos;
				new_move.capture = target_figure;
				moves[*i] = new_move;
                inc(i);
			}
		}
		
		// 1.2 middle
		target_pos = pos - 1;
		if((target_figure = b->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(target_figure) != IS_BLACK(figure))
			{
				new_move.capture = target_figure;
				new_move.to = target_pos;
				captures[*i] = new_move;
                inc(i);
			}
		}
		else
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves[*i] = new_move;
            inc(i);
		}
		
		// 1.3 down
		if(row > 0)
		{
			target_pos = pos - 9;
			if((target_figure = b->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(target_figure) != IS_BLACK(figure))
				{
					new_move.capture = target_figure;
					new_move.to = target_pos;
					captures[*i] = new_move;
                    inc(i);
				}
			}
			else
			{
				new_move.to = target_pos;
				new_move.capture = target_figure;
				moves[*i] = new_move;
                inc(i);
			}
		}
	}
	
	// 2. Move right
	if(col < 7)
	{
		// 2.1 up
		if(row < 7)
		{
			target_pos = pos + 9;
			if((target_figure = b->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(target_figure) != IS_BLACK(figure))
				{
					new_move.capture = target_figure;
					new_move.to = target_pos;
					captures[*i] = new_move;
                    inc(i);
				}
			}
			else
			{
				new_move.to = target_pos;
				new_move.capture = target_figure;
				moves[*i] = new_move;
                inc(i);
			}
		}
		
		// 2.2 middle
		target_pos = pos + 1;
		if((target_figure = b->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(target_figure) != IS_BLACK(figure))
			{
				new_move.capture = target_figure;
				new_move.to = target_pos;
				captures[*i] = new_move;
                inc(i);
			}
		}
		else
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves[*i] = new_move;
            inc(i);
		}
		
		// 2.3 down
		if(row > 0)
		{
			target_pos = pos - 7;
			if((target_figure = b->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(target_figure) != IS_BLACK(figure))
				{
					new_move.capture = target_figure;
					new_move.to = target_pos;
					captures[*i] = new_move;
                    inc(i);
				}
			}
			else
			{
				new_move.to = target_pos;
				new_move.capture = target_figure;
				moves[*i] = new_move;
                inc(i);
			}
		}
	}
	
	// 3. straight up
	if(row < 7)
	{
		// 2.2 middle
		target_pos = pos + 8;
		if((target_figure = b->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(target_figure) != IS_BLACK(figure))
			{
				new_move.capture = target_figure;
				new_move.to = target_pos;
				captures[*i] = new_move;
                inc(i);
			}
		}
		else
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves[*i] = new_move;
            inc(i);
		}	
	}
	
	// 4. straight down
	if(row > 0)
	{
		// 2.2 middle
		target_pos = pos - 8;
		if((target_figure = b->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(target_figure) != IS_BLACK(figure))
			{
				new_move.capture = target_figure;
				new_move.to = target_pos;
				captures[*i] = new_move;
                inc(i);
			}
		}
		else
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves[*i] = new_move;
            inc(i);
		}	
	}

	// 5. Castling
	if(!IS_MOVED(figure) && !isVulnerable_C(b, pos, figure))
	{
		// short
		target_pos = IS_BLACK(figure) ? F8 : F1;
		if((b->square[target_pos] == EMPTY) && !isVulnerable_C(b, target_pos, figure))
		{
			target_pos = IS_BLACK(figure) ? G8 : G1;
			if((b->square[target_pos] == EMPTY) && !isVulnerable_C(b, target_pos, figure))
			{
				target_pos = IS_BLACK(figure) ? H8 : H1;
				target_figure = b->square[target_pos];
				if(!IS_MOVED(target_figure) && (FIGURE(target_figure) == ROOK) && !isVulnerable_C(b, target_pos, figure))
				{
					if(IS_BLACK(target_figure) == IS_BLACK(figure))
					{
						new_move.capture = EMPTY;
						new_move.to = IS_BLACK(figure) ? G8 : G1;
						moves[*i] = new_move;
                        inc(i);
					}
				}
			}
		}
		
		// long
		target_pos = IS_BLACK(figure) ? B8 : B1;
		if((b->square[target_pos] == EMPTY) && !isVulnerable_C(b, target_pos, figure))
		{
			target_pos = IS_BLACK(figure) ? C8 : C1;
			if((b->square[target_pos] == EMPTY) && !isVulnerable_C(b, target_pos, figure))
			{
				target_pos = IS_BLACK(figure) ? D8 : D1;
				if((b->square[target_pos] == EMPTY) && !isVulnerable_C(b, target_pos, figure))
				{
					target_pos = IS_BLACK(figure) ? A8 : A1;
					target_figure = b->square[target_pos];
					if(!IS_MOVED(target_figure) && (FIGURE(target_figure) == ROOK) && !isVulnerable_C(b, target_pos, figure))
					{
						if(IS_BLACK(target_figure) == IS_BLACK(figure))
						{
							new_move.capture = EMPTY;
							new_move.to = IS_BLACK(figure) ? C8 : C1;
							moves[*i] = new_move;
                            inc(i);
						}
					}
				}
			}
		}
	}
}

__device__ void getMoves_C(ChessBoard *b, int color, 
        Move *moves, 
        Move *captures, int *i)
{
	int pos, figure;
    *i = 0;
	
	for(pos = 0; pos < 64; pos++)
	{
		if((figure = b->square[pos]) != EMPTY)
		{
			if(IS_BLACK(figure) == color)
			{
				switch(FIGURE(figure))
				{
					case PAWN:
						getPawnMoves_C(b, figure, pos, moves, captures, i);
						break;
					case ROOK:
						getRookMoves_C(b, figure, pos, moves, captures, i);
						break;
					case KNIGHT:
						getKnightMoves_C(b, figure, pos, moves, captures, i);
						break;
					case BISHOP:
						getBishopMoves_C(b, figure, pos, moves, captures, i);
						break;
					case QUEEN:
						getQueenMoves_C(b, figure, pos, moves, captures, i);
						break;
					case KING:
						getKingMoves_C(b, figure, pos, moves, captures, i);
						break;
					default:
						break;
				}
			}
		}
	}
}

__device__ void movePawn_C(ChessBoard *b, const Move & move)
{
	int capture_field;

	// check for en-passant capture
	if(IS_PASSANT(move.capture))
	{
		if(IS_BLACK(move.figure))
		{
			capture_field = move.to + 8;
			if((move.from / 8) == 3)
				b->square[capture_field] = EMPTY;
		}
		else
		{
			capture_field = move.to - 8;
			if((move.from / 8) == 4)
				b->square[capture_field] = EMPTY;
		}
	}

	b->square[(int)move.from] = EMPTY;

	// mind pawn promotion
	if(IS_BLACK(move.figure)) {
		if(move.to / 8 == 0)
			b->square[(int)move.to] = SET_MOVED(SET_BLACK(QUEEN));
		else
			b->square[(int)move.to] = SET_MOVED(move.figure);
	}
	else {
		if(move.to / 8 == 7)
			b->square[(int)move.to] = SET_MOVED(QUEEN);
		else
			b->square[(int)move.to] = SET_MOVED(move.figure);
	}
}

__device__ void undoMovePawn_C(ChessBoard *b, const Move & move)
{
	int capture_field;

	b->square[(int)move.from] = CLEAR_PASSANT(move.figure);

	// check for en-passant capture
	if(IS_PASSANT(move.capture))
	{
		if(IS_BLACK(move.figure))
		{
			capture_field = move.to + 8;
			if(move.from / 8 == 3) {
				b->square[capture_field] = move.capture;
				b->square[(int)move.to] = EMPTY;
			}
			else {
				b->square[(int)move.to] = move.capture;
			}
		}
		else
		{
			capture_field = move.to - 8;
			if(move.from / 8 == 4) {
				b->square[capture_field] = move.capture;
				b->square[(int)move.to] = EMPTY;
			}
			else {
				b->square[(int)move.to] = move.capture;
			}
		}
	}
	else
	{
		b->square[(int)move.to] = move.capture;
	}
}

__device__ void moveKing_C(ChessBoard *b, const Move & move)
{
	// check for castling
	if(!IS_MOVED(move.figure))
	{
		switch(move.to)
		{
			case G1:
				b->square[H1] = EMPTY;
				b->square[F1] = SET_MOVED(ROOK);
				break;
			case G8:
				b->square[H8] = EMPTY;
				b->square[F8] = SET_MOVED(SET_BLACK(ROOK));
				break;
			case C1:
				b->square[A1] = EMPTY;
				b->square[D1] = SET_MOVED(ROOK);
				break;
			case C8:
				b->square[A8] = EMPTY;
				b->square[D8] = SET_MOVED(SET_BLACK(ROOK));
				break;
			default:
				break;
		}
	}

	// regular move
	b->square[(int)move.from] = EMPTY;
	b->square[(int)move.to] = SET_MOVED(move.figure);
	
	// update king position variable
	if(IS_BLACK(move.figure)) {
		b->black_king_pos = move.to;
	}
	else {
		b->white_king_pos = move.to;
	}
}

__device__ void undoMoveKing_C(ChessBoard *b, const Move & move)
{
	// check for castling
	if(!IS_MOVED(move.figure))
	{
		// set rook depending on
		// king's target field
		switch(move.to)
		{
			case G1:
				b->square[H1] = ROOK;
				b->square[F1] = EMPTY;
				break;
			case G8:
				b->square[H8] = SET_BLACK(ROOK);
				b->square[F8] = EMPTY;
				break;
			case C1:
				b->square[A1] = ROOK;
				b->square[D1] = EMPTY;
				break;
			case C8:
				b->square[A8] = SET_BLACK(ROOK);
				b->square[D8] = EMPTY;
				break;
			default:
				break;
		}
	}

	// regular undo
	b->square[(int)move.from] = move.figure;
	b->square[(int)move.to] = move.capture;

	// update king position variable
	if(IS_BLACK(move.figure)) {
		b->black_king_pos = move.from;
	}
	else {
		b->white_king_pos = move.from;
	}
}

__device__ void move_C(ChessBoard *b, const Move & move)
{
	// kings and pawns receive special treatment
	switch(FIGURE(move.figure))
	{
		case KING:
			moveKing_C(b, move);
			break;
		case PAWN:
			if(move.to != move.from) {
				movePawn_C(b, move);
				break;
			}
		default:
			b->square[(int)move.from] = EMPTY;
			b->square[(int)move.to] = SET_MOVED(move.figure);
			break;
	}
}

__device__ void undoMove_C(ChessBoard *b, const Move & move)
{
	// kings and pawns receive special treatment
	switch(FIGURE(move.figure))
	{
		case KING:
			undoMoveKing_C(b, move);
			break;
		case PAWN:
			if(move.to != move.from) {
				undoMovePawn_C(b, move);
				break;
			}
		default:
			b->square[(int)move.from] = move.figure;
			b->square[(int)move.to] = move.capture;
			break;
	}
}

__device__ bool isValidMove_C(ChessBoard *b, int color, Move move)
{
	bool valid = false;
	Move *regulars = (Move *)malloc(sizeof(Move)*MSIZE);
    int z = 0;

	getMoves_C(b, color, regulars, regulars, &z);

	for(int j = 0; j < z && !valid; j++)
	{
		if(move.from == regulars[j].from && move.to == regulars[j].to)
		{
			move = regulars[j];

			move_C(b, move);
			if(!isVulnerable_C(b, color ? b->black_king_pos : b->white_king_pos, color))
				valid = true;
			undoMove_C(b, regulars[j]);
		}
	}
    free(regulars);
	return valid;
}

__device__ void apply_move_C(State *s, Move *move) {
    if(!(s->white_to_move)) s->depth--;
    s->white_to_move = !(s->white_to_move);
    ChessBoard b = s->board;
    move_C(&b, *move);
    s->board = b;
}

__device__ __inline__ float
max_cuda(float a, float b) {
    return a<b?b:a;
}

__device__ __inline__ float
min_cuda(float a, float b) {
    return a>b?b:a;
}

__device__ float minimax_cuda(State *state, int index){
    //malloc/allocate Move array of length 100
    //printf("Beginning minimax, %d\n", index);

    Move *move = (Move *)malloc(sizeof(Move)*MSIZE);
    int i = 0;
   
    if(move == NULL) printf("Out of room\n");
 
    ChessBoard *b = (ChessBoard *)malloc(sizeof(ChessBoard));
    *b = state->board;
    bool kvuln, can_move = false;
    bool isterm = false;

    
    int color = state->get_color();
    getMoves_C(b, color, move, move, &i);
    if(isVulnerable_C(b, color ? b->black_king_pos : b->white_king_pos, color))
        kvuln = true;
    for(int j = 0; j < i && !can_move; j++) {
        move_C(b, move[j]);
        if(!isVulnerable_C(b, color ? b->black_king_pos : b->white_king_pos, color))
            can_move = true;
        undoMove_C(b, move[j]);
    }
    //Checkmate or stalemate
    if(!can_move) isterm = true;
    if(state->depth == 0 && state->white_to_move) isterm = true;
    if(isterm) {
        //if checkmate, return VICTORY, else LOSS
        //printf("Returning %d\n", index);
        free(move);
        free(b);
        if(!can_move && kvuln) return VICTORY;
        else return LOSS;
    }

    //printf("Depth %d, index:%d cont\n", state.depth, index);

    if(!state->white_to_move) {
        float value = -INF;
        State *next_state = (State*)malloc(sizeof(State));
        for(int j = 0; j < i; j++) {
            *next_state = *state;
            *b = state->board;
            if(!isValidMove_C(b, color, move[j])) continue;
            apply_move_C(next_state, &move[j]);
            
            //printf("CUDA check 3.0, size: %d\n", i);

            value = max_cuda(value, minimax_cuda(next_state, index));
            if(value == VICTORY) {
                free(move);
                free(b);
                free(next_state);
                return value;
            }
        }
        free(move);
        free(b);
        free(next_state);
        return LOSS;
    }
    else {
        float value = INF;
        State *next_state = (State*)malloc(sizeof(State));
        for(int j = 0; j < i; j++) {
            *next_state = *state;
            *b = state->board;
            if(!isValidMove_C(b, color, move[j])) continue;
            apply_move_C(next_state, &move[j]);

            //printf("CUDA check 3\n");
            
            value = min_cuda(value, minimax_cuda(next_state, index));
            if(value == LOSS) {
                free(b);
                free(move);
                free(next_state);
                return value;
            }
        }
        free(move);
        free(b);
        free(next_state);
        return VICTORY;
    }
}

__global__ void
minim_kernel(State *s, float* res, int *len) {
    // get State and Action corresponding to index
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    /*if(index == 0) {
        printf("Length: %d\n", *len);
    }*/

    if(index >= *len) return;
    //Assume one state

    //res[index] = 3.0;
    //return;
    
    State state = s[index];
    
    /*if(!(state.depth == 1 && !state.white_to_move))
        printf("values: %d, %d\n", state.depth, int(state.white_to_move));*/
    
    Move *move = (Move *)malloc(sizeof(Move)*MSIZE);
    if(move == NULL) printf("Out of space\n");
    int i = 0;    
    ChessBoard *b = (ChessBoard *)malloc(sizeof(ChessBoard));
    *b = state.board;
    bool kvuln, can_move = false;
    bool isterm = false;

    
    int color = state.get_color();
    getMoves_C(b, color, move, move, &i);
    
    //printf("Test: %d\n", int(move[i].from));
    if(i == 0) printf("Value: %d\n", i);
    
    if(isVulnerable_C(b, color ? b->black_king_pos : b->white_king_pos, color))
        kvuln = true;
    for(int j = 0; j < i && !can_move; j++) {
        move_C(b, move[j]);
        if(!isVulnerable_C(b, color ? b->black_king_pos : b->white_king_pos, color))
            can_move = true;
        undoMove_C(b, move[j]);
    }
    //Checkmate or stalemate
    if(!can_move) isterm = true;
    if(state.depth == 0 && state.white_to_move) isterm = true;
    if(isterm) {
        //if checkmate, return VICTORY, else LOSS
        free(move);
        free(b);
        if(!can_move && kvuln) res[index] = VICTORY;
        else res[index] = LOSS;
        //if(index == 0) printf("Successful return pt 2\n");
        return;
    }

    //if(index == 0) printf("Start\n");

    if(!state.white_to_move) {
        //printf("Check %d\n", index);

        float value = -INF;
        for(int j = 0; j < i; j++) {
            State next_state = state;
            *b = state.board;
            
            if(!isValidMove_C(b, color, move[j])){
                //printf("Skipped %d %d\n", index, j);
                continue;
            }

            //printf("Check2.0 %d %d %d\n", index, j, i);

            apply_move_C(&next_state, &move[j]);

            //printf("Check3.0 %d %d %d\n", index, j, i);
            
            float val = minimax_cuda(&next_state, index);
            
            //printf("Checkout %d %d %d\n", index, j, i);
            
            value = max_cuda(value, val);
            
            //printf("Check result %d %d\n", index, j);
            
            if(value == VICTORY) {
                free(move);
                free(b);
                res[index] = value;
                return;
            }
        }
        res[index] = LOSS;
    }
    else {
        //printf("Check %d\n", index);
        float value = INF;
        for(int j = 0; j < i; j++) {
            State next_state = state;
            *b = state.board;
            if(!isValidMove_C(b, color, move[j])) continue;

            //printf("Check2 %d\n", index);

            apply_move_C(&next_state, &move[j]);

            //printf("Check3 %d\n", index);
            
            value = min_cuda(value, minimax_cuda(&next_state, index));
            if(value == LOSS) {
                free(move);
                free(b);
                res[index] = value;
                return;
            }
        }
        res[index] = VICTORY;
    }
    free(move);
    free(b);
}

float mini_Rec_CUDA(State state, bool *set) {
    State* s;
    float* res;
    float* resultarray;
    int len;

    const int threadsPerBlock = 64;

    std::vector<Action> actions;
    state.get_actions(actions);
    len = actions.size();
    const int blocks = (len + threadsPerBlock - 1)/threadsPerBlock;
    
    if(state.is_terminal())
        return state.evaluate_minimax();
    if(state.depth == 2 && !state.white_to_move) {
        //Build array of States
        State *st = (State*)malloc(sizeof(State)*actions.size());
        for(int i = 0; i < actions.size(); i++) {
            st[i] = state;
            st[i].apply_action(actions.at(i));
            /*if(!(st[i].depth == 1 && !st[i].white_to_move))
                printf("isTerm test inputs wrong!\n");*/
        }
        
        int numState = actions.size();
        int len = actions.size();
        
        //call CUDA kernel to evaluate_minimax
        if(!state.white_to_move){
            float value = -INF;
            int *l;
            cudaError_t errCode;
            resultarray = (float*)malloc(sizeof(float)*len);
            // execute kernel
            /*for(int i = 0; i < len; i++) {
                printf("Initial Value %d: %f\n", i, resultarray[i]);
            }*/
            
            /*if(st[0].depth == 0 && st[0].white_to_move)
                printf("isTerm test inputs correct\n");*/
            
            // pass values to GPU

            cudaMalloc((void**)&s, sizeof(State) * numState);
            errCode = cudaPeekAtLastError();
            if (errCode != cudaSuccess) {
                fprintf(stderr, "WARNING: A CUDA error cudaMalloc s \
not white to move: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
                exit(1744);
            }
            cudaMalloc((void**)&res, sizeof(float) * len);
            errCode = cudaPeekAtLastError();
            if (errCode != cudaSuccess) {
                fprintf(stderr, "WARNING: A CUDA error cudaMalloc res \
not white to move: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
                exit(1751);
            }
            cudaMalloc((void**)&l, sizeof(int));
            errCode = cudaPeekAtLastError();
            if (errCode != cudaSuccess) {
                fprintf(stderr, "WARNING: A CUDA error cudaMalloc l \
not white to move: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
                exit(1758);
            }

            cudaCheckError(cudaMemcpy(s, (void *)st, sizeof(State)*numState, 
                    cudaMemcpyHostToDevice));
            errCode = cudaPeekAtLastError();
            if (errCode != cudaSuccess) {
                fprintf(stderr, "WARNING: A CUDA error cudaMemcpy s \
not white to move: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
                exit(1767);
            }
            cudaCheckError(cudaMemcpy(l, (void *)&len, sizeof(int), cudaMemcpyHostToDevice));
            
            if(!(*set)) {
                cudaDeviceSetLimit(cudaLimitMallocHeapSize, 12*1024*1024);
                *set = !(*set);
            }
            errCode = cudaPeekAtLastError();
            if (errCode != cudaSuccess) {
                fprintf(stderr, "WARNING: A CUDA error occured before kernel launch with \
not white to move: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
                exit(1776);
            }

            // execute kernel
            minim_kernel<<<blocks, threadsPerBlock>>>(s, res, l);

            errCode = cudaPeekAtLastError();
            if (errCode != cudaSuccess) {
                fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
                exit(1785);
            }
            //printf("Successful return1.0\n");

            cudaCheckError(cudaDeviceSynchronize());

            cudaCheckError(cudaMemcpy(resultarray, res, sizeof(float)*len, 
                    cudaMemcpyDeviceToHost));

            //free values
            cudaFree(s);
            cudaFree(res);
            cudaFree(l);

            //use resultarray
            errCode = cudaPeekAtLastError();
            if (errCode != cudaSuccess) {
                fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
                exit(1803);
            }
            //printf("Successful return\n");
            free(st);
            for(int i = 0; i < len; i++) {
                value = max(value, resultarray[i]);
                if(value == VICTORY) {
                    free(resultarray);
                    //printf("Found result, %d\n", i);
                    return VICTORY;
                }
                //printf("Value %d: %f\n", i, resultarray[i]);
            }
            free(resultarray);

            //value = max(value, minimax(next_state));
            //if(value == VICTORY)
                //return Action(it->regular, it->nulls, value);
            return LOSS;
        }
        else {
            printf("White to move\n");
            float value = INF;
            int *l;

            cudaError_t errCode;
     
            resultarray = (float*)malloc(sizeof(float)*len);
             // execute kernel
             // pass values to GPU
            cudaMalloc((void**)&s, sizeof(State) * numState);
            cudaMalloc((void**)&res, sizeof(float) * len);
            cudaMalloc((void**)&l, sizeof(int));

            cudaMemcpy(s, (void*)st, sizeof(State)*numState,
                    cudaMemcpyHostToDevice);
            cudaMemcpy(l, (void*)&len, sizeof(int), cudaMemcpyHostToDevice);
             // execute kernel
             
            if(!(*set)) {
                cudaDeviceSetLimit(cudaLimitMallocHeapSize, 12*1024*1024);
                *set = !(*set);
            }
             
            minim_kernel<<<blocks, threadsPerBlock>>>(s, res, l);
            
            errCode = cudaPeekAtLastError();
            if (errCode != cudaSuccess) {
                fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
                exit(1846);
            }
     
            cudaCheckError(cudaDeviceSynchronize());
             
            cudaMemcpy(resultarray, res, sizeof(float)*len,
                    cudaMemcpyDeviceToHost);
             
            //free values
            cudaFree(s);
            cudaFree(res);
            cudaFree(l);
            //use resultarray
            
            free(st);
            //remove nulls?
            for(int i = 0; i < len; i++) {
                value = min(value, resultarray[i]);
                if(value == LOSS) {
                  free(resultarray);
                  return LOSS;
                }
            }
            free(resultarray);
            return VICTORY;
        }
    }    

    if(!state.white_to_move){
        float value = -INF;
        for(std::vector<Action>::iterator it=actions.begin(); it!=actions.end(); it++) {
            State next_state = state;
            next_state.apply_action(*it);
            value = max(value, mini_Rec_CUDA(next_state, set));
            if(value == VICTORY) {
                printf("Found path to VIC\n");
                return value;
            }
        } 
        return LOSS;
    }
    else {
        float value = INF;
        for(std::vector<Action>::iterator it=actions.begin(); it!=actions.end(); it++) {
            State next_state = state;
            next_state.apply_action(*it);
            value = min(value, mini_Rec_CUDA(next_state, set));
            if(value == LOSS) return value;
        } 
        printf("No path to LOSS\n");
        return VICTORY;
    }
}

Action
minimaxCuda(State state, bool *set) {
    State* s;
    int numState = 1;
    float* res;
    float* resultarray;
    int len;

    const int threadsPerBlock = 64;

    std::vector<Action> actions;
    state.get_actions(actions);
    len = actions.size();
    const int blocks = (len + threadsPerBlock - 1)/threadsPerBlock;

    if(state.is_terminal())
        return Action(state.evaluate_minimax());
    if(state.depth == 2 && !state.white_to_move) {
        //Build array of States
        State *st = (State*)malloc(sizeof(State)*actions.size());
        for(int i = 0; i < actions.size(); i++) {
            st[i] = state;
            st[i].apply_action(actions.at(i));
        }
        
        numState = len;
        
        //call CUDA kernel to evaluate_minimax
        if(!state.white_to_move){
            float value = -INF;
            int *l;
            cudaError_t errCode;
            resultarray = (float*)malloc(sizeof(float)*len);
            // execute kernel
            // pass values to GPU

            cudaMalloc((void**)&s, sizeof(State) * numState);
            cudaMalloc((void**)&res, sizeof(float) * len);
            cudaMalloc((void**)&l, sizeof(int));

            cudaCheckError(cudaMemcpy(s, (void *)st, sizeof(State)*numState, 
                    cudaMemcpyHostToDevice));
            cudaCheckError(cudaMemcpy(l, (void *)&len, sizeof(int), cudaMemcpyHostToDevice));
            
            if(!(*set)) {
                cudaDeviceSetLimit(cudaLimitMallocHeapSize, 12*1024*1024);
                *set = !(*set);
            }
            errCode = cudaPeekAtLastError();
            if (errCode != cudaSuccess) {
                fprintf(stderr, "WARNING: A CUDA error occured before k launch: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
                exit(1948);
            }

            // execute kernel
            minim_kernel<<<blocks, threadsPerBlock>>>(s, res, l);

            errCode = cudaPeekAtLastError();
            if (errCode != cudaSuccess) {
                fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
                exit(1957);
            }

            cudaCheckError(cudaDeviceSynchronize());

            cudaCheckError(cudaMemcpy(resultarray, res, sizeof(float)*len, 
                    cudaMemcpyDeviceToHost));

            //free values
            cudaFree(s);
            cudaFree(res);
            cudaFree(l);

            //use resultarray
            errCode = cudaPeekAtLastError();
            if (errCode != cudaSuccess) {
                fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
                exit(1974);
            }
            
            free(st);
            for(int i = 0; i < len; i++) {
                value = max(value, resultarray[i]);
                if(value == VICTORY) {
                    free(resultarray);
                    return Action(actions.at(i).regular, actions.at(i).nulls, VICTORY);
                }
            }
            free(resultarray);

            //value = max(value, minimax(next_state));
            //if(value == VICTORY)
                //return Action(it->regular, it->nulls, value);
            return Action(actions.begin()->regular, actions.begin()->nulls, LOSS);
        }
        else {
            printf("White to move\n");
            float value = INF;
            int *l;

            cudaError_t errCode;
     
            resultarray = (float*)malloc(sizeof(float)*len);
             // execute kernel
             // pass values to GPU
            cudaMalloc((void**)&s, sizeof(State) * numState);
            cudaMalloc((void**)&res, sizeof(float) * len);
            cudaMalloc((void**)&l, sizeof(int));

            cudaMemcpy(s, (void*)st, sizeof(State)*numState,
                    cudaMemcpyHostToDevice);
            cudaMemcpy(l, (void*)&len, sizeof(int), cudaMemcpyHostToDevice);
             // execute kernel
            
            if(!(*set)) {
                cudaDeviceSetLimit(cudaLimitMallocHeapSize, 12*1024*1024);
                *set = !(*set);
            }            
            
            minim_kernel<<<blocks, threadsPerBlock>>>(s, res, l);
            
            errCode = cudaPeekAtLastError();
            if (errCode != cudaSuccess) {
                fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
                exit(2017);
            }
     
            cudaCheckError(cudaDeviceSynchronize());
             
            cudaMemcpy(resultarray, res, sizeof(float)*len,
                    cudaMemcpyDeviceToHost);
             
            //free values
            cudaFree(s);
            cudaFree(res);
            cudaFree(l);
            //use resultarray
            
            free(st);
            //remove nulls?
            for(int i = 0; i < len; i++) {
                value = min(value, resultarray[i]);
                if(value == LOSS) {
                  free(resultarray);
                  return Action(actions.at(i).regular, actions.at(i).nulls, LOSS);
                }
            }
            free(resultarray);
            return Action(actions.begin()->regular, actions.begin()->nulls, VICTORY);
        }
    }
    if(!state.white_to_move){
        float value = -INF;
        for(std::vector<Action>::iterator it=actions.begin(); it!=actions.end(); it++) {
            State next_state = state;
            next_state.apply_action(*it);
            value = max(value, mini_Rec_CUDA(next_state, set));
            if(value == VICTORY) {
                printf("VICTORY\n");
                return Action(it->regular, it->nulls, VICTORY);
            }
        } 
        return Action(actions.begin()->regular, actions.begin()->nulls, LOSS);;
    }
    else {
        float value = INF;
        for(std::vector<Action>::iterator it=actions.begin(); it!=actions.end(); it++) {
            State next_state = state;
            next_state.apply_action(*it);
            value = min(value, mini_Rec_CUDA(next_state, set));
            if(value == LOSS) {
                return Action(it->regular, it->nulls, LOSS);
            }
        } 
        return Action(actions.begin()->regular, actions.begin()->nulls, VICTORY);;
    }
}
}
}
