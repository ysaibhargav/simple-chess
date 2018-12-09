#include <cstdio>
#include <cstring>
#include <string>
#include <sstream>
#include <vector>
#include <iterator>
#include <iostream>
#include <list>
#include "chessboard.h"
#include "chessplayer.h"

//#include <thrust/device_vector.h>

#define MSIZE 100

#ifdef __CUDA_ARCH__
#define DEV __device__
#define HOST __host__
#else
#define DEV
#define HOST
#endif

using namespace std;

void Move::print(void) const {

	const char * field_name[] = {
		"A1", "B1", "C1", "D1", "E1", "F1", "G1", "H1",
		"A2", "B2", "C2", "D2", "E2", "F2", "G2", "H2",
		"A3", "B3", "C3", "D3", "E3", "F3", "G3", "H3",
		"A4", "B4", "C4", "D4", "E4", "F4", "G4", "H4",
		"A5", "B5", "C5", "D5", "E5", "F5", "G5", "H5",
		"A6", "B6", "C6", "D6", "E6", "F6", "G6", "H6",
		"A7", "B7", "C7", "D7", "E7", "F7", "G7", "H7",
		"A8", "B8", "C8", "D8", "E8", "F8", "G8", "H8"
	};		

	if(IS_BLACK(figure))
		printf("   Black ");
	else 
		printf("   White ");

	switch(FIGURE(figure)) {
		case PAWN:
			printf("pawn ");
			break;
		case ROOK:
			printf("rook ");
			break;
		case KNIGHT:
			printf("knight ");
			break;
		case BISHOP:
			printf("bishop ");
			break;
		case QUEEN:
			printf("queen ");
			break;
		case KING:
			printf("king ");
			break;
	}
	
	printf("from %s to %s:\n", field_name[(int)from], field_name[(int)to]);
}

bool Move::operator==(const Move & b) const
{
	if(from != b.from)
		return false;
	if(to != b.to)
		return false;
	if(capture != b.capture)
		return false;
	if(figure != b.figure)
		return false;
		
	return true;
}


ChessBoard::ChessBoard()
{
	memset((void*)square, EMPTY, sizeof(square));
}

void ChessBoard::print(void) const
{
	char figure;
	int repr, unmoved, passant, row, col;

	printf("   ___ ___ ___ ___ ___ ___ ___ ___ \n  ");

	for(row = 7; row >= 0; row--)
	{
		for(col = 0; col < 8; col++)
		{
			figure = this->square[row*8+col];
			repr = getASCIIrepr(figure);
			unmoved = (IS_MOVED(figure) || (figure == EMPTY)) ? ' ' : '.';
			passant = IS_PASSANT(figure) ? '`' : ' ';
			printf("|%c%c%c", unmoved, repr, passant);
		}
		printf("|\n%d |___|___|___|___|___|___|___|___|\n  ", row + 1);
	}
	printf("  A   B   C   D   E   F   G   H  \n\n");
}

char ChessBoard::getASCIIrepr(int figure) const
{
	switch(FIGURE(figure))
	{
		case PAWN:
			if(IS_BLACK(figure))
				return 'o';
			else
				return 'x';
		case ROOK:
			if(IS_BLACK(figure))
				return 'T';
			else
				return 't';
		case KNIGHT:
			if(IS_BLACK(figure))
				return 'H';
			else
				return'h';
		case BISHOP:
			if(IS_BLACK(figure))
				return 'B';
			else
				return 'b';
		case QUEEN:
			if(IS_BLACK(figure))
				return 'Q';
			else
				return 'q';
		case KING:
			if(IS_BLACK(figure))
				return 'K';
			else
				return 'k';
	}
	
	return ' ';
}

void ChessBoard::initFENSetup(std::string FEN)
{
    //clear board
    memset((void*)square, EMPTY, sizeof(square));

    //Split FEN over spaces
    std::istringstream iss(FEN);
    std::vector<std::string> results(std::istream_iterator<std::string>{iss},
        std::istream_iterator<std::string>());

    std::string board = results[0];
    //Split FEN[0] over /

    //Set castling (SETmoved)
    std::string cast = results[2];
    std::size_t fK = cast.find('k');
    std::size_t fQ = cast.find('q');
    char bk = SET_BLACK(KING);
    if(fK == std::string::npos && fQ == std::string::npos) {
        bk = SET_MOVED(SET_BLACK(KING));
        //cout << "MOVED";
    }
    std::size_t fk = cast.find('K');
    std::size_t fq = cast.find('Q');
    char wk = KING;
    if(fk == std::string::npos && fq == std::string::npos)
        wk = SET_MOVED(KING);

    char brk = SET_BLACK(ROOK);
    char brq = SET_BLACK(ROOK);
    if(fK == std::string::npos)
        brk = SET_MOVED(brk);
    if(fQ == std::string::npos)
        brq = SET_MOVED(brq);
    char wrk = ROOK;
    char wrq = ROOK;
    if(fk == std::string::npos)
        wrk = SET_MOVED(wrk);
    if(fq == std::string::npos)
        wrq = SET_MOVED(wrq);
    //int sq = 1;
    //char row = 'A';
    int start = 56;
    for(unsigned int i = 0; i < board.length(); i++) {
        char c = board[i];
        if(isdigit(c)) {
            int step = c - '0';
            //sq += step;
            start += step;
        }
        else {
            if(c == '/') {
                //row += 1;
                //sq = 1;
                //start += 1;
                start -= 16;
            }
            else {
                //std::string loc;
                //loc << row << sq;
                switch(c) {
                    case 'p':
                        if(start >= 48 && start < 56)
                            square[start] = SET_BLACK(PAWN);
                        else
                            square[start] = SET_MOVED(SET_BLACK(PAWN));
                        break;
                    case 'n':
                        if(start == 57 || start == 62)
                            square[start] = SET_BLACK(KNIGHT);
                        else
                            square[start] = SET_MOVED(SET_BLACK(KNIGHT));
                        break;
                    case 'b':
                        if(start == 58 || start == 61)
                            square[start] = SET_BLACK(BISHOP);
                        else
                            square[start] = SET_MOVED(SET_BLACK(BISHOP));
                        break;
                    case 'r':
                        if(start == 56)
                            square[start] = brq;
                        else if(start == 63)
                            square[start] = brk;
                        else
                            square[start] = SET_MOVED(SET_BLACK(ROOK));
                        break;
                    case 'q':
                        if(start == 59)
                            square[start] = SET_BLACK(QUEEN);
                        else
                            square[start] = SET_MOVED(SET_BLACK(QUEEN));
                        break;
                    case 'k':
                        square[start] = bk;
                        black_king_pos = start;
                        break;
                    case 'P':
                        if(start >= 8 && start < 16)
                            square[start] = PAWN;
                        else
                            square[start] = SET_MOVED(PAWN);
                        break;
                    case 'R':
                        if(start == 0)
                            square[start] = wrq;
                        else if(start == 7)
                            square[start] = wrk;
                        else
                            square[start] = SET_MOVED(ROOK);
                        break;
                    case 'N':
                        if(start == 1 || start == 6)
                            square[start] = KNIGHT;
                        else
                            square[start] = SET_MOVED(KNIGHT);
                        break;
                    case 'B':
                        if(start == 2 || start == 5)
                            square[start] = BISHOP;
                        else
                            square[start] = SET_MOVED(BISHOP);
                        break;
                    case 'Q':
                        if(start == 3)
                            square[start] = QUEEN;
                        else
                            square[start] = SET_MOVED(QUEEN);
                        break;
                    case 'K':
                        square[start] = wk;
                        white_king_pos = start;
                        break;
                }
                //sq++;
                start++;
            }
        }
    }
    //Set next to move!
    //Set castling (SETmoved)
    /*string cast = results[2];
    std::size_t fK = cast.find('K');
    std::size_t fQ = cast.find('Q');
    if(fK == std::string::npos && fQ == std::string::npos)
        char bk = SET_MOVED(SET_BLACK(KING));
    std::size_t fk = cast.find('k');
    std::size_t fq = cast.find('q');
    if(fk == std::string::npos && fq == std::string::npos)
        char wk = SET_MOVED(KING);*/
}

void ChessBoard::initDefaultSetup(void)
{
	// clear board
	memset((void*)square, EMPTY, sizeof(square));

	// setup white aristocracy
	square[A1] = ROOK; square[B1] = KNIGHT; square[C1] = BISHOP; square[D1] = QUEEN;
	square[E1] = KING; square[F1] = BISHOP; square[G1] = KNIGHT; square[H1] = ROOK;
	
	// setup black aristocracy
	square[A8] = SET_BLACK(ROOK); square[B8] = SET_BLACK(KNIGHT);
	square[C8] = SET_BLACK(BISHOP); square[D8] = SET_BLACK(QUEEN);
	square[E8] = SET_BLACK(KING); square[F8] = SET_BLACK(BISHOP);
	square[G8] = SET_BLACK(KNIGHT); square[H8] = SET_BLACK(ROOK);
	
	// setup white pawns
	square[A2] = square[B2] = square[C2] = square[D2] =
		square[E2] = square[F2] = square[G2] = square[H2] = PAWN;
		
	// setup black pawns
	square[A7] = square[B7] = square[C7] = square[D7] =
		square[E7] = square[F7] = square[G7] = square[H7] = SET_BLACK(PAWN);

	// register kings
	black_king_pos = E8;
	white_king_pos = E1;
}

void ChessBoard::getMoves(int color, list<Move> & moves, list<Move> & captures, list<Move> & null_moves)
{
	int pos, figure;
	
	for(pos = 0; pos < 64; pos++)
	{
		if((figure = this->square[pos]) != EMPTY)
		{
			if(IS_BLACK(figure) == color)
			{
				switch(FIGURE(figure))
				{
					case PAWN:
						getPawnMoves(figure, pos, moves, captures, null_moves);
						break;
					case ROOK:
						getRookMoves(figure, pos, moves, captures);
						break;
					case KNIGHT:
						getKnightMoves(figure, pos, moves, captures);
						break;
					case BISHOP:
						getBishopMoves(figure, pos, moves, captures);
						break;
					case QUEEN:
						getQueenMoves(figure, pos, moves, captures);
						break;
					case KING:
						getKingMoves(figure, pos, moves, captures);
						break;
					default:
						break;
				}
			}
		}
	}
}

void ChessBoard::getPawnMoves(int figure, int pos, list<Move> & moves, list<Move> & captures, list<Move>  & null_moves) const
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
		null_moves.push_back(new_move);
		
		figure = CLEAR_PASSANT(figure);
	}

	// Of course, we only have to set this once
	new_move.figure = figure;
	new_move.from = pos;

	// 1. One step ahead
	target_pos = IS_BLACK(figure) ? pos - 8 : pos + 8;
	if((target_pos >= 0) && (target_pos < 64))
	{
		if((target_figure = this->square[target_pos]) == EMPTY)
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves.push_back(new_move);
			
			// 2. Two steps ahead if unmoved
			if(!IS_MOVED(figure))
			{
				target_pos = IS_BLACK(figure) ? pos - 16 : pos + 16;
				if((target_pos >= 0) && (target_pos < 64))
				{
					if((target_figure = this->square[target_pos]) == EMPTY)
					{
						new_move.to = target_pos;
						new_move.capture = target_figure;

						// set passant attribute and clear it later
						new_move.figure = SET_PASSANT(figure);
						moves.push_back(new_move);
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
			if((target_figure = this->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(target_figure) != IS_BLACK(figure))
				{
					new_move.to = target_pos;
					new_move.capture = target_figure;
					captures.push_back(new_move);
				}
			}
			else
			{
				// En passant?
				target_figure = this->square[pos - 1];
				if(IS_PASSANT(target_figure))
				{
					if(IS_BLACK(target_figure) != IS_BLACK(figure))
					{
						new_move.to = target_pos;
						new_move.capture = target_figure;
						captures.push_back(new_move);
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
			if((target_figure = this->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(target_figure) != IS_BLACK(figure))
				{
					new_move.to = target_pos;
					new_move.capture = target_figure;
					captures.push_back(new_move);
				}
			}
			else
			{
				// En passant?
				target_figure = this->square[pos + 1];
				if(IS_PASSANT(target_figure))
				{
					if(IS_BLACK(target_figure) != IS_BLACK(figure))
					{
						new_move.to = target_pos;
						new_move.capture = target_figure;
						captures.push_back(new_move);
					}				
				}
			}
		}
	}	
}

void ChessBoard::getRookMoves(int figure, int pos, list<Move> & moves, list<Move> & captures) const
{
	Move new_move;
	int target_pos, target_figure, end;

	// Of course, we only have to set this once
	new_move.figure = figure;
	new_move.from = pos;

	// 1. Move up
	for(target_pos = pos + 8; target_pos < 64; target_pos += 8)
	{
		if((target_figure = this->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(figure) != IS_BLACK(target_figure))
			{
				new_move.to = target_pos;
				new_move.capture = target_figure;
				captures.push_back(new_move);
			}
			
			break;
		}
		else
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves.push_back(new_move);
		}
	}

	// 2. Move down
	for(target_pos = pos - 8; target_pos >= 0; target_pos -= 8)
	{	
		if((target_figure = this->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(figure) != IS_BLACK(target_figure))
			{
				new_move.to = target_pos;
				new_move.capture = target_figure;
				captures.push_back(new_move);
			}
			
			break;
		}
		else
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves.push_back(new_move);
		}
	}

	// 3. Move left
	for(target_pos = pos - 1, end = pos - (pos % 8); target_pos >= end; target_pos--)
	{
		if((target_figure = this->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(figure) != IS_BLACK(target_figure))
			{
				new_move.to = target_pos;
				new_move.capture = target_figure;
				captures.push_back(new_move);
			}
			
			break;
		}
		else
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves.push_back(new_move);
		}
	}

	// 4. Move right
	for(target_pos = pos + 1, end = pos + (8 - pos % 8); target_pos < end; target_pos++)
	{
		if((target_figure = this->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(figure) != IS_BLACK(target_figure))
			{
				new_move.to = target_pos;
				new_move.capture = target_figure;
				captures.push_back(new_move);
			}
			
			break;
		}
		else
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves.push_back(new_move);
		}
	}
}

void ChessBoard::getKnightMoves(int figure, int pos, list<Move> & moves, list<Move> & captures) const
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
		
			if((target_figure = this->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(figure) != IS_BLACK(target_figure))
				{
					new_move.capture = target_figure;
					new_move.to = target_pos;
					captures.push_back(new_move);
				}
			}
			else
			{
				new_move.capture = target_figure;
				new_move.to = target_pos;
				moves.push_back(new_move);
			}
		}
		
		// left
		if(col > 0)
		{
			target_pos = pos + 15;
		
			if((target_figure = this->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(figure) != IS_BLACK(target_figure))
				{
					new_move.capture = target_figure;
					new_move.to = target_pos;
					captures.push_back(new_move);
				}
			}
			else
			{
				new_move.capture = target_figure;
				new_move.to = target_pos;
				moves.push_back(new_move);
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
		
			if((target_figure = this->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(figure) != IS_BLACK(target_figure))
				{
					new_move.capture = target_figure;
					new_move.to = target_pos;
					captures.push_back(new_move);
				}
			}
			else
			{
				new_move.capture = target_figure;
				new_move.to = target_pos;
				moves.push_back(new_move);
			}
		}
		
		// left
		if(col > 0)
		{
			target_pos = pos - 17;
		
			if((target_figure = this->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(figure) != IS_BLACK(target_figure))
				{
					new_move.capture = target_figure;
					new_move.to = target_pos;
					captures.push_back(new_move);
				}
			}
			else
			{
				new_move.capture = target_figure;
				new_move.to = target_pos;
				moves.push_back(new_move);
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
		
			if((target_figure = this->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(figure) != IS_BLACK(target_figure))
				{
					new_move.capture = target_figure;
					new_move.to = target_pos;
					captures.push_back(new_move);
				}
			}
			else
			{
				new_move.capture = target_figure;
				new_move.to = target_pos;
				moves.push_back(new_move);
			}
		}
		
		// down
		if(row > 0)
		{
			target_pos = pos - 6;
		
			if((target_figure = this->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(figure) != IS_BLACK(target_figure))
				{
					new_move.capture = target_figure;
					new_move.to = target_pos;
					captures.push_back(new_move);
				}
			}
			else
			{
				new_move.capture = target_figure;
				new_move.to = target_pos;
				moves.push_back(new_move);
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
		
			if((target_figure = this->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(figure) != IS_BLACK(target_figure))
				{
					new_move.capture = target_figure;
					new_move.to = target_pos;
					captures.push_back(new_move);
				}
			}
			else
			{
				new_move.capture = target_figure;
				new_move.to = target_pos;
				moves.push_back(new_move);
			}
		}
		
		// down
		if(row > 0)
		{
			target_pos = pos - 10;
		
			if((target_figure = this->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(figure) != IS_BLACK(target_figure))
				{
					new_move.capture = target_figure;
					new_move.to = target_pos;
					captures.push_back(new_move);
				}
			}
			else
			{
				new_move.capture = target_figure;
				new_move.to = target_pos;
				moves.push_back(new_move);
			}		
		}
	}	
}

void ChessBoard::getBishopMoves(int figure, int pos, list<Move> & moves, list<Move> & captures) const
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
		if((target_figure = this->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(figure) != IS_BLACK(target_figure))
			{
				new_move.to = target_pos;
				new_move.capture = target_figure;
				captures.push_back(new_move);
			}

			break;
		}
		else
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves.push_back(new_move);
		}
	}
	
	// 2. Go south-east
	for(i = row - 1, j = col + 1; (i >= 0) && (j < 8); i--, j++)
	{
		target_pos = i * 8 + j;
		if((target_figure = this->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(figure) != IS_BLACK(target_figure))
			{
				new_move.to = target_pos;
				new_move.capture = target_figure;
				captures.push_back(new_move);
			}
			
			break;
		}
		else
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves.push_back(new_move);
		}
	}

	// 3. Go south-west
	for(i = row - 1, j = col - 1; (i >= 0) && (j >= 0); i--, j--)
	{
		target_pos = i * 8 + j;
		if((target_figure = this->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(figure) != IS_BLACK(target_figure))
			{
				new_move.to = target_pos;
				new_move.capture = target_figure;
				captures.push_back(new_move);
			}
			
			break;
		}
		else
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves.push_back(new_move);
		}
	}

	// 4. Go north-west
	for(i = row + 1, j = col - 1; (i < 8) && (j >= 0); i++, j--)
	{
		target_pos = i * 8 + j;
		if((target_figure = this->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(figure) != IS_BLACK(target_figure))
			{
				new_move.to = target_pos;
				new_move.capture = target_figure;
				captures.push_back(new_move);
			}
			
			break;
		}
		else
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves.push_back(new_move);
		}
	}
}

void ChessBoard::getQueenMoves(int figure, int pos, list<Move> & moves, list<Move> & captures) const
{
	// Queen is just the "cartesian product" of Rook and Bishop
	this->getRookMoves(figure, pos, moves, captures);
	this->getBishopMoves(figure, pos, moves, captures);
}


void ChessBoard::getKingMoves(int figure, int pos, list<Move> & moves, list<Move> & captures)
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
			if((target_figure = this->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(target_figure) != IS_BLACK(figure))
				{
					new_move.capture = target_figure;
					new_move.to = target_pos;
					captures.push_back(new_move);
				}
			}
			else
			{
				new_move.to = target_pos;
				new_move.capture = target_figure;
				moves.push_back(new_move);
			}
		}
		
		// 1.2 middle
		target_pos = pos - 1;
		if((target_figure = this->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(target_figure) != IS_BLACK(figure))
			{
				new_move.capture = target_figure;
				new_move.to = target_pos;
				captures.push_back(new_move);
			}
		}
		else
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves.push_back(new_move);
		}
		
		// 1.3 down
		if(row > 0)
		{
			target_pos = pos - 9;
			if((target_figure = this->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(target_figure) != IS_BLACK(figure))
				{
					new_move.capture = target_figure;
					new_move.to = target_pos;
					captures.push_back(new_move);
				}
			}
			else
			{
				new_move.to = target_pos;
				new_move.capture = target_figure;
				moves.push_back(new_move);
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
			if((target_figure = this->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(target_figure) != IS_BLACK(figure))
				{
					new_move.capture = target_figure;
					new_move.to = target_pos;
					captures.push_back(new_move);
				}
			}
			else
			{
				new_move.to = target_pos;
				new_move.capture = target_figure;
				moves.push_back(new_move);
			}
		}
		
		// 2.2 middle
		target_pos = pos + 1;
		if((target_figure = this->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(target_figure) != IS_BLACK(figure))
			{
				new_move.capture = target_figure;
				new_move.to = target_pos;
				captures.push_back(new_move);
			}
		}
		else
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves.push_back(new_move);
		}
		
		// 2.3 down
		if(row > 0)
		{
			target_pos = pos - 7;
			if((target_figure = this->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(target_figure) != IS_BLACK(figure))
				{
					new_move.capture = target_figure;
					new_move.to = target_pos;
					captures.push_back(new_move);
				}
			}
			else
			{
				new_move.to = target_pos;
				new_move.capture = target_figure;
				moves.push_back(new_move);
			}
		}
	}
	
	// 3. straight up
	if(row < 7)
	{
		// 2.2 middle
		target_pos = pos + 8;
		if((target_figure = this->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(target_figure) != IS_BLACK(figure))
			{
				new_move.capture = target_figure;
				new_move.to = target_pos;
				captures.push_back(new_move);
			}
		}
		else
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves.push_back(new_move);
		}	
	}
	
	// 4. straight down
	if(row > 0)
	{
		// 2.2 middle
		target_pos = pos - 8;
		if((target_figure = this->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(target_figure) != IS_BLACK(figure))
			{
				new_move.capture = target_figure;
				new_move.to = target_pos;
				captures.push_back(new_move);
			}
		}
		else
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves.push_back(new_move);
		}	
	}

	// 5. Castling
	if(!IS_MOVED(figure) && !isVulnerable(pos, figure))
	{
		// short
		target_pos = IS_BLACK(figure) ? F8 : F1;
		if((this->square[target_pos] == EMPTY) && !isVulnerable(target_pos, figure))
		{
			target_pos = IS_BLACK(figure) ? G8 : G1;
			if((this->square[target_pos] == EMPTY) && !isVulnerable(target_pos, figure))
			{
				target_pos = IS_BLACK(figure) ? H8 : H1;
				target_figure = this->square[target_pos];
				if(!IS_MOVED(target_figure) && (FIGURE(target_figure) == ROOK) && !isVulnerable(target_pos, figure))
				{
					if(IS_BLACK(target_figure) == IS_BLACK(figure))
					{
						new_move.capture = EMPTY;
						new_move.to = IS_BLACK(figure) ? G8 : G1;
						moves.push_back(new_move);
					}
				}
			}
		}
		
		// long
		target_pos = IS_BLACK(figure) ? B8 : B1;
		if((this->square[target_pos] == EMPTY) && !isVulnerable(target_pos, figure))
		{
			target_pos = IS_BLACK(figure) ? C8 : C1;
			if((this->square[target_pos] == EMPTY) && !isVulnerable(target_pos, figure))
			{
				target_pos = IS_BLACK(figure) ? D8 : D1;
				if((this->square[target_pos] == EMPTY) && !isVulnerable(target_pos, figure))
				{
					target_pos = IS_BLACK(figure) ? A8 : A1;
					target_figure = this->square[target_pos];
					if(!IS_MOVED(target_figure) && (FIGURE(target_figure) == ROOK) && !isVulnerable(target_pos, figure))
					{
						if(IS_BLACK(target_figure) == IS_BLACK(figure))
						{
							new_move.capture = EMPTY;
							new_move.to = IS_BLACK(figure) ? C8 : C1;
							moves.push_back(new_move);
						}
					}
				}
			}
		}
	}
}

DEV void ChessBoard::getMoves_cuda(int color, 
        Move *moves, 
        Move *captures, int *i)
{
	int pos, figure;
    *i = 0;
	
	for(pos = 0; pos < 64; pos++)
	{
		if((figure = this->square[pos]) != EMPTY)
		{
			if(IS_BLACK(figure) == color)
			{
				switch(FIGURE(figure))
				{
					case PAWN:
						getPawnMoves_cuda(figure, pos, moves, captures, i);
						break;
					case ROOK:
						getRookMoves_cuda(figure, pos, moves, captures, i);
						break;
					case KNIGHT:
						getKnightMoves_cuda(figure, pos, moves, captures, i);
						break;
					case BISHOP:
						getBishopMoves_cuda(figure, pos, moves, captures, i);
						break;
					case QUEEN:
						getQueenMoves_cuda(figure, pos, moves, captures, i);
						break;
					case KING:
						getKingMoves_cuda(figure, pos, moves, captures, i);
						break;
					default:
						break;
				}
			}
		}
	}
}

DEV void ChessBoard::getPawnMoves_cuda(int figure, int pos, 
        Move *moves, 
        Move *captures, int *i) const
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
		if((target_figure = this->square[target_pos]) == EMPTY)
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves[*i] = new_move;
            (*i)++;
			
			// 2. Two steps ahead if unmoved
			if(!IS_MOVED(figure))
			{
				target_pos = IS_BLACK(figure) ? pos - 16 : pos + 16;
				if((target_pos >= 0) && (target_pos < 64))
				{
					if((target_figure = this->square[target_pos]) == EMPTY)
					{
						new_move.to = target_pos;
						new_move.capture = target_figure;

						// set passant attribute and clear it later
						new_move.figure = SET_PASSANT(figure);
						moves[*i] = new_move;
                        (*i)++;
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
			if((target_figure = this->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(target_figure) != IS_BLACK(figure))
				{
					new_move.to = target_pos;
					new_move.capture = target_figure;
					captures[*i] = new_move;
                    (*i)++;
				}
			}
			else
			{
				// En passant?
				target_figure = this->square[pos - 1];
				if(IS_PASSANT(target_figure))
				{
					if(IS_BLACK(target_figure) != IS_BLACK(figure))
					{
						new_move.to = target_pos;
						new_move.capture = target_figure;
						captures[*i] = new_move;
                        (*i)++;
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
			if((target_figure = this->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(target_figure) != IS_BLACK(figure))
				{
					new_move.to = target_pos;
					new_move.capture = target_figure;
					captures[*i] = new_move;
                    (*i)++;
				}
			}
			else
			{
				// En passant?
				target_figure = this->square[pos + 1];
				if(IS_PASSANT(target_figure))
				{
					if(IS_BLACK(target_figure) != IS_BLACK(figure))
					{
						new_move.to = target_pos;
						new_move.capture = target_figure;
						captures[*i] = new_move;
                        (*i)++;
					}				
				}
			}
		}
	}	
}

DEV void ChessBoard::getRookMoves_cuda(int figure, int pos, 
        Move *moves, 
        Move *captures, int *i) const
{
	Move new_move;
	int target_pos, target_figure, end;

	// Of course, we only have to set this once
	new_move.figure = figure;
	new_move.from = pos;

	// 1. Move up
	for(target_pos = pos + 8; target_pos < 64; target_pos += 8)
	{
		if((target_figure = this->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(figure) != IS_BLACK(target_figure))
			{
				new_move.to = target_pos;
				new_move.capture = target_figure;
				captures[*i] = new_move;
                (*i)++;
			}
			
			break;
		}
		else
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves[*i] = new_move;
            (*i)++;
		}
	}

	// 2. Move down
	for(target_pos = pos - 8; target_pos >= 0; target_pos -= 8)
	{	
		if((target_figure = this->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(figure) != IS_BLACK(target_figure))
			{
				new_move.to = target_pos;
				new_move.capture = target_figure;
				captures[*i] = new_move;
                (*i)++;
			}
			
			break;
		}
		else
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves[*i] = new_move;
            (*i)++;
		}
	}

	// 3. Move left
	for(target_pos = pos - 1, end = pos - (pos % 8); target_pos >= end; target_pos--)
	{
		if((target_figure = this->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(figure) != IS_BLACK(target_figure))
			{
				new_move.to = target_pos;
				new_move.capture = target_figure;
				captures[*i] = new_move;
                (*i)++;
			}
			
			break;
		}
		else
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves[*i] = new_move;
            (*i)++;
		}
	}

	// 4. Move right
	for(target_pos = pos + 1, end = pos + (8 - pos % 8); target_pos < end; target_pos++)
	{
		if((target_figure = this->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(figure) != IS_BLACK(target_figure))
			{
				new_move.to = target_pos;
				new_move.capture = target_figure;
				captures[*i] = new_move;
                (*i)++;
			}
			
			break;
		}
		else
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves[*i] = new_move;
            (*i)++;
		}
	}
}

DEV void ChessBoard::getKnightMoves_cuda(int figure, int pos, 
        Move *moves, 
        Move *captures, int *i) const
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
		
			if((target_figure = this->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(figure) != IS_BLACK(target_figure))
				{
					new_move.capture = target_figure;
					new_move.to = target_pos;
					captures[*i] = new_move;
                    (*i)++;
				}
			}
			else
			{
				new_move.capture = target_figure;
				new_move.to = target_pos;
				moves[*i] = new_move;
                (*i)++;
			}
		}
		
		// left
		if(col > 0)
		{
			target_pos = pos + 15;
		
			if((target_figure = this->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(figure) != IS_BLACK(target_figure))
				{
					new_move.capture = target_figure;
					new_move.to = target_pos;
					captures[*i] = new_move;
                    (*i)++;
				}
			}
			else
			{
				new_move.capture = target_figure;
				new_move.to = target_pos;
				moves[*i] = new_move;
                (*i)++;
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
		
			if((target_figure = this->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(figure) != IS_BLACK(target_figure))
				{
					new_move.capture = target_figure;
					new_move.to = target_pos;
					captures[*i] = new_move;
                    (*i)++;
				}
			}
			else
			{
				new_move.capture = target_figure;
				new_move.to = target_pos;
				moves[*i] = new_move;
                (*i)++;
			}
		}
		
		// left
		if(col > 0)
		{
			target_pos = pos - 17;
		
			if((target_figure = this->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(figure) != IS_BLACK(target_figure))
				{
					new_move.capture = target_figure;
					new_move.to = target_pos;
					captures[*i] = new_move;
                    (*i)++;
				}
			}
			else
			{
				new_move.capture = target_figure;
				new_move.to = target_pos;
				moves[*i] = new_move;
                (*i)++;
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
		
			if((target_figure = this->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(figure) != IS_BLACK(target_figure))
				{
					new_move.capture = target_figure;
					new_move.to = target_pos;
					captures[*i] = new_move;
                    (*i)++;
				}
			}
			else
			{
				new_move.capture = target_figure;
				new_move.to = target_pos;
				moves[*i] = new_move;
                (*i)++;
			}
		}
		
		// down
		if(row > 0)
		{
			target_pos = pos - 6;
		
			if((target_figure = this->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(figure) != IS_BLACK(target_figure))
				{
					new_move.capture = target_figure;
					new_move.to = target_pos;
					captures[*i] = new_move;
                    (*i)++;
				}
			}
			else
			{
				new_move.capture = target_figure;
				new_move.to = target_pos;
				moves[*i] = new_move;
                (*i)++;
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
		
			if((target_figure = this->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(figure) != IS_BLACK(target_figure))
				{
					new_move.capture = target_figure;
					new_move.to = target_pos;
					captures[*i] = new_move;
                    (*i)++;
				}
			}
			else
			{
				new_move.capture = target_figure;
				new_move.to = target_pos;
				moves[*i] = new_move;
                (*i)++;
			}
		}
		
		// down
		if(row > 0)
		{
			target_pos = pos - 10;
		
			if((target_figure = this->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(figure) != IS_BLACK(target_figure))
				{
					new_move.capture = target_figure;
					new_move.to = target_pos;
					captures[*i] = new_move;
                    (*i)++;
				}
			}
			else
			{
				new_move.capture = target_figure;
				new_move.to = target_pos;
				moves[*i] = new_move;
                (*i)++;
			}		
		}
	}	
}

DEV void ChessBoard::getBishopMoves_cuda(int figure, int pos, 
        Move *moves, 
        Move *captures, int *loc) const
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
		if((target_figure = this->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(figure) != IS_BLACK(target_figure))
			{
				new_move.to = target_pos;
				new_move.capture = target_figure;
				captures[*loc] = new_move;
                (*loc)++;
			}

			break;
		}
		else
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves[*loc] = new_move;
            (*loc)++;
		}
	}
	
	// 2. Go south-east
	for(i = row - 1, j = col + 1; (i >= 0) && (j < 8); i--, j++)
	{
		target_pos = i * 8 + j;
		if((target_figure = this->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(figure) != IS_BLACK(target_figure))
			{
				new_move.to = target_pos;
				new_move.capture = target_figure;
				captures[*loc] = new_move;
                (*loc)++;
			}
			
			break;
		}
		else
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves[*loc] = new_move;
            (*loc)++;
		}
	}

	// 3. Go south-west
	for(i = row - 1, j = col - 1; (i >= 0) && (j >= 0); i--, j--)
	{
		target_pos = i * 8 + j;
		if((target_figure = this->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(figure) != IS_BLACK(target_figure))
			{
				new_move.to = target_pos;
				new_move.capture = target_figure;
				captures[*loc] = new_move;
                (*loc)++;
			}
			
			break;
		}
		else
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves[*loc] = new_move;
            (*loc)++;
		}
	}

	// 4. Go north-west
	for(i = row + 1, j = col - 1; (i < 8) && (j >= 0); i++, j--)
	{
		target_pos = i * 8 + j;
		if((target_figure = this->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(figure) != IS_BLACK(target_figure))
			{
				new_move.to = target_pos;
				new_move.capture = target_figure;
				captures[*loc] = new_move;
                (*loc)++;
			}
			
			break;
		}
		else
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves[*loc] = new_move;
            (*loc)++;
		}
	}
}

DEV void ChessBoard::getQueenMoves_cuda(int figure, int pos, Move *moves, 
        Move *captures, int *i) const
{
	// Queen is just the "cartesian product" of Rook and Bishop
	this->getRookMoves_cuda(figure, pos, moves, captures, i);
	this->getBishopMoves_cuda(figure, pos, moves, captures, i);
}


DEV void ChessBoard::getKingMoves_cuda(int figure, int pos, Move *moves,
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
			if((target_figure = this->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(target_figure) != IS_BLACK(figure))
				{
					new_move.capture = target_figure;
					new_move.to = target_pos;
					captures[*i] = new_move;
                    (*i)++;
				}
			}
			else
			{
				new_move.to = target_pos;
				new_move.capture = target_figure;
				moves[*i] = new_move;
                (*i)++;
			}
		}
		
		// 1.2 middle
		target_pos = pos - 1;
		if((target_figure = this->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(target_figure) != IS_BLACK(figure))
			{
				new_move.capture = target_figure;
				new_move.to = target_pos;
				captures[*i] = new_move;
                (*i)++;
			}
		}
		else
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves[*i] = new_move;
            (*i)++;
		}
		
		// 1.3 down
		if(row > 0)
		{
			target_pos = pos - 9;
			if((target_figure = this->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(target_figure) != IS_BLACK(figure))
				{
					new_move.capture = target_figure;
					new_move.to = target_pos;
					captures[*i] = new_move;
                    (*i)++;
				}
			}
			else
			{
				new_move.to = target_pos;
				new_move.capture = target_figure;
				moves[*i] = new_move;
                (*i)++;
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
			if((target_figure = this->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(target_figure) != IS_BLACK(figure))
				{
					new_move.capture = target_figure;
					new_move.to = target_pos;
					captures[*i] = new_move;
                    (*i)++;
				}
			}
			else
			{
				new_move.to = target_pos;
				new_move.capture = target_figure;
				moves[*i] = new_move;
                (*i)++;
			}
		}
		
		// 2.2 middle
		target_pos = pos + 1;
		if((target_figure = this->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(target_figure) != IS_BLACK(figure))
			{
				new_move.capture = target_figure;
				new_move.to = target_pos;
				captures[*i] = new_move;
                (*i)++;
			}
		}
		else
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves[*i] = new_move;
            (*i)++;
		}
		
		// 2.3 down
		if(row > 0)
		{
			target_pos = pos - 7;
			if((target_figure = this->square[target_pos]) != EMPTY)
			{
				if(IS_BLACK(target_figure) != IS_BLACK(figure))
				{
					new_move.capture = target_figure;
					new_move.to = target_pos;
					captures[*i] = new_move;
                    (*i)++;
				}
			}
			else
			{
				new_move.to = target_pos;
				new_move.capture = target_figure;
				moves[*i] = new_move;
                (*i)++;
			}
		}
	}
	
	// 3. straight up
	if(row < 7)
	{
		// 2.2 middle
		target_pos = pos + 8;
		if((target_figure = this->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(target_figure) != IS_BLACK(figure))
			{
				new_move.capture = target_figure;
				new_move.to = target_pos;
				captures[*i] = new_move;
                (*i)++;
			}
		}
		else
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves[*i] = new_move;
            (*i)++;
		}	
	}
	
	// 4. straight down
	if(row > 0)
	{
		// 2.2 middle
		target_pos = pos - 8;
		if((target_figure = this->square[target_pos]) != EMPTY)
		{
			if(IS_BLACK(target_figure) != IS_BLACK(figure))
			{
				new_move.capture = target_figure;
				new_move.to = target_pos;
				captures[*i] = new_move;
                (*i)++;
			}
		}
		else
		{
			new_move.to = target_pos;
			new_move.capture = target_figure;
			moves[*i] = new_move;
            (*i)++;
		}	
	}

	// 5. Castling
	if(!IS_MOVED(figure) && !isVulnerable(pos, figure))
	{
		// short
		target_pos = IS_BLACK(figure) ? F8 : F1;
		if((this->square[target_pos] == EMPTY) && !isVulnerable(target_pos, figure))
		{
			target_pos = IS_BLACK(figure) ? G8 : G1;
			if((this->square[target_pos] == EMPTY) && !isVulnerable(target_pos, figure))
			{
				target_pos = IS_BLACK(figure) ? H8 : H1;
				target_figure = this->square[target_pos];
				if(!IS_MOVED(target_figure) && (FIGURE(target_figure) == ROOK) && !isVulnerable(target_pos, figure))
				{
					if(IS_BLACK(target_figure) == IS_BLACK(figure))
					{
						new_move.capture = EMPTY;
						new_move.to = IS_BLACK(figure) ? G8 : G1;
						moves[*i] = new_move;
                        (*i)++;
					}
				}
			}
		}
		
		// long
		target_pos = IS_BLACK(figure) ? B8 : B1;
		if((this->square[target_pos] == EMPTY) && !isVulnerable(target_pos, figure))
		{
			target_pos = IS_BLACK(figure) ? C8 : C1;
			if((this->square[target_pos] == EMPTY) && !isVulnerable(target_pos, figure))
			{
				target_pos = IS_BLACK(figure) ? D8 : D1;
				if((this->square[target_pos] == EMPTY) && !isVulnerable(target_pos, figure))
				{
					target_pos = IS_BLACK(figure) ? A8 : A1;
					target_figure = this->square[target_pos];
					if(!IS_MOVED(target_figure) && (FIGURE(target_figure) == ROOK) && !isVulnerable(target_pos, figure))
					{
						if(IS_BLACK(target_figure) == IS_BLACK(figure))
						{
							new_move.capture = EMPTY;
							new_move.to = IS_BLACK(figure) ? C8 : C1;
							moves[*i] = new_move;
                            (*i)++;
						}
					}
				}
			}
		}
	}
}

HOST DEV bool ChessBoard::isVulnerable(int pos, int figure) const
{
	int target_pos, target_figure, row, col, i,j, end;

	// Determine row and column
	row = pos / 8;
	col = pos % 8;

	// 1. Look for Rooks, Queens and Kings above
	for(target_pos = pos + 8; target_pos < 64; target_pos += 8)
	{
		if((target_figure = this->square[target_pos]) != EMPTY)
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
		if((target_figure = this->square[target_pos]) != EMPTY)
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
		if((target_figure = this->square[target_pos]) != EMPTY)
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
		if((target_figure = this->square[target_pos]) != EMPTY)
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
		if((target_figure = this->square[target_pos]) != EMPTY)
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
		if((target_figure = this->square[target_pos]) != EMPTY)
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
		if((target_figure = this->square[target_pos]) != EMPTY)
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
		if((target_figure = this->square[target_pos]) != EMPTY)
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

			if((target_figure = this->square[target_pos]) != EMPTY)
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
		
			if((target_figure = this->square[target_pos]) != EMPTY)
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
		
			if((target_figure = this->square[target_pos]) != EMPTY)
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
		
			if((target_figure = this->square[target_pos]) != EMPTY)
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
		
			if((target_figure = this->square[target_pos]) != EMPTY)
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
		
			if((target_figure = this->square[target_pos]) != EMPTY)
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
		
			if((target_figure = this->square[target_pos]) != EMPTY)
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
		
			if((target_figure = this->square[target_pos]) != EMPTY)
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

bool ChessBoard::isValidMove(int color, Move & move)
{
	bool valid = false;
	list<Move> regulars, nulls;

	getMoves(color, regulars, regulars, nulls);

	for(list<Move>::iterator it = regulars.begin(); it != regulars.end() && !valid; ++it)
	{
		if(move.from == (*it).from && move.to == (*it).to)
		{
			move = *it;

			this->move(move);
			if(!isVulnerable(color ? black_king_pos : white_king_pos, color))
				valid = true;
			undoMove(*it);
		}
	}

	return valid;
}

DEV bool ChessBoard::isValidMove_cuda(int color, Move & move)
{
	bool valid = false;
	Move regulars[MSIZE];
    int i = 0;

	getMoves_cuda(color, regulars, regulars, &i);

	for(int j = 0; j < i && !valid; j++)
	{
		if(move.from == regulars[j].from && move.to == regulars[j].to)
		{
			move = regulars[j];

			this->move(move);
			if(!isVulnerable(color ? black_king_pos : white_king_pos, color))
				valid = true;
			undoMove(regulars[j]);
		}
	}

	return valid;
}

ChessPlayer::Status ChessBoard::getPlayerStatus(int color)
{
	bool king_vulnerable = false, can_move = false;
	list<Move> regulars, nulls;

	getMoves(color, regulars, regulars, nulls);

	if(isVulnerable(color ? black_king_pos : white_king_pos, color))
		king_vulnerable = true;

	for(list<Move>::iterator it = regulars.begin(); it != regulars.end() && !can_move; ++it)
	{
		this->move(*it);
		if(!isVulnerable(color ? black_king_pos : white_king_pos, color))
		{
			can_move = true;
		}
		undoMove(*it);
	}

	if(king_vulnerable && can_move)
		return ChessPlayer::InCheck;
	if(king_vulnerable && !can_move)
		return ChessPlayer::Checkmate;
	if(!king_vulnerable && !can_move)
		return ChessPlayer::Stalemate;

	return ChessPlayer::Normal;
}

DEV ChessPlayer::Status ChessBoard::getPlayerStatus_cuda(int color,
        Move *act, bool writeflag, int *i)
{
	bool king_vulnerable = false, can_move = false;
	//Action *nulls;

    if(writeflag) {
	    getMoves_cuda(color, act, act, i);
    }

	if(isVulnerable(color ? black_king_pos : white_king_pos, color))
		king_vulnerable = true;

    //loop length
	for(int j = 0; j < *i && !can_move; j++)
	{
		this->move(act[j]);
		if(!isVulnerable(color ? black_king_pos : white_king_pos, color))
		{
			can_move = true;
		}
		undoMove(act[j]);
	}

	if(king_vulnerable && can_move)
		return ChessPlayer::InCheck;
	if(king_vulnerable && !can_move)
		return ChessPlayer::Checkmate;
	if(!king_vulnerable && !can_move)
		return ChessPlayer::Stalemate;

	return ChessPlayer::Normal;
}

HOST DEV void ChessBoard::move(const Move & move)
{
	// kings and pawns receive special treatment
	switch(FIGURE(move.figure))
	{
		case KING:
			moveKing(move);
			break;
		case PAWN:
			if(move.to != move.from) {
				movePawn(move);
				break;
			}
		default:
			this->square[(int)move.from] = EMPTY;
			this->square[(int)move.to] = SET_MOVED(move.figure);
			break;
	}
}

HOST DEV void ChessBoard::undoMove(const Move & move)
{
	// kings and pawns receive special treatment
	switch(FIGURE(move.figure))
	{
		case KING:
			undoMoveKing(move);
			break;
		case PAWN:
			if(move.to != move.from) {
				undoMovePawn(move);
				break;
			}
		default:
			this->square[(int)move.from] = move.figure;
			this->square[(int)move.to] = move.capture;
			break;
	}
}

HOST DEV void ChessBoard::movePawn(const Move & move)
{
	int capture_field;

	// check for en-passant capture
	if(IS_PASSANT(move.capture))
	{
		if(IS_BLACK(move.figure))
		{
			capture_field = move.to + 8;
			if((move.from / 8) == 3)
				this->square[capture_field] = EMPTY;
		}
		else
		{
			capture_field = move.to - 8;
			if((move.from / 8) == 4)
				this->square[capture_field] = EMPTY;
		}
	}

	this->square[(int)move.from] = EMPTY;

	// mind pawn promotion
	if(IS_BLACK(move.figure)) {
		if(move.to / 8 == 0)
			this->square[(int)move.to] = SET_MOVED(SET_BLACK(QUEEN));
		else
			this->square[(int)move.to] = SET_MOVED(move.figure);
	}
	else {
		if(move.to / 8 == 7)
			this->square[(int)move.to] = SET_MOVED(QUEEN);
		else
			this->square[(int)move.to] = SET_MOVED(move.figure);
	}
}

HOST DEV void ChessBoard::undoMovePawn(const Move & move)
{
	int capture_field;

	this->square[(int)move.from] = CLEAR_PASSANT(move.figure);

	// check for en-passant capture
	if(IS_PASSANT(move.capture))
	{
		if(IS_BLACK(move.figure))
		{
			capture_field = move.to + 8;
			if(move.from / 8 == 3) {
				this->square[capture_field] = move.capture;
				this->square[(int)move.to] = EMPTY;
			}
			else {
				this->square[(int)move.to] = move.capture;
			}
		}
		else
		{
			capture_field = move.to - 8;
			if(move.from / 8 == 4) {
				this->square[capture_field] = move.capture;
				this->square[(int)move.to] = EMPTY;
			}
			else {
				this->square[(int)move.to] = move.capture;
			}
		}
	}
	else
	{
		this->square[(int)move.to] = move.capture;
	}
}

HOST DEV void ChessBoard::moveKing(const Move & move)
{
	// check for castling
	if(!IS_MOVED(move.figure))
	{
		switch(move.to)
		{
			case G1:
				this->square[H1] = EMPTY;
				this->square[F1] = SET_MOVED(ROOK);
				break;
			case G8:
				this->square[H8] = EMPTY;
				this->square[F8] = SET_MOVED(SET_BLACK(ROOK));
				break;
			case C1:
				this->square[A1] = EMPTY;
				this->square[D1] = SET_MOVED(ROOK);
				break;
			case C8:
				this->square[A8] = EMPTY;
				this->square[D8] = SET_MOVED(SET_BLACK(ROOK));
				break;
			default:
				break;
		}
	}

	// regular move
	this->square[(int)move.from] = EMPTY;
	this->square[(int)move.to] = SET_MOVED(move.figure);
	
	// update king position variable
	if(IS_BLACK(move.figure)) {
		black_king_pos = move.to;
	}
	else {
		white_king_pos = move.to;
	}
}

HOST DEV void ChessBoard::undoMoveKing(const Move & move)
{
	// check for castling
	if(!IS_MOVED(move.figure))
	{
		// set rook depending on
		// king's target field
		switch(move.to)
		{
			case G1:
				this->square[H1] = ROOK;
				this->square[F1] = EMPTY;
				break;
			case G8:
				this->square[H8] = SET_BLACK(ROOK);
				this->square[F8] = EMPTY;
				break;
			case C1:
				this->square[A1] = ROOK;
				this->square[D1] = EMPTY;
				break;
			case C8:
				this->square[A8] = SET_BLACK(ROOK);
				this->square[D8] = EMPTY;
				break;
			default:
				break;
		}
	}

	// regular undo
	this->square[(int)move.from] = move.figure;
	this->square[(int)move.to] = move.capture;

	// update king position variable
	if(IS_BLACK(move.figure)) {
		black_king_pos = move.from;
	}
	else {
		white_king_pos = move.from;
	}
}
