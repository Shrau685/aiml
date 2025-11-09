import java.util.Scanner;

public class EasyTicTacToe {
    public static void main(String[] args) {
        // Make a 3x3 board
        char[] board = {'1','2','3','4','5','6','7','8','9'};
        char turn = 'X'; // X goes first
        Scanner sc = new Scanner(System.in);

        while (true) {
            printBoard(board);
            System.out.println("Player " + turn + ", choose a number (1-9): ");
            int move = sc.nextInt();

            // Check if that place is free
            if (board[move - 1] == 'X' || board[move - 1] == 'O') {
                System.out.println("Oops! That spot is taken. Try again!");
                continue;
            }

            // Place X or O
            board[move - 1] = turn;

            // Check if someone won
            if (checkWin(board)) {
                printBoard(board);
                System.out.println(" Player " + turn + " wins!");
                break;
            }

            // Check if board is full
            if (isFull(board)) {
                printBoard(board);
                System.out.println("It's a draw!");
                break;
            }

            // Switch turns
            turn = (turn == 'X') ? 'O' : 'X';
        }
        sc.close();
    }

    // Print the board in a nice way
    static void printBoard(char[] b) {
        System.out.println();
        System.out.println(" " + b[0] + " | " + b[1] + " | " + b[2]);
        System.out.println("---+---+---");
        System.out.println(" " + b[3] + " | " + b[4] + " | " + b[5]);
        System.out.println("---+---+---");
        System.out.println(" " + b[6] + " | " + b[7] + " | " + b[8]);
        System.out.println();
    }

    // Check all winning lines
    static boolean checkWin(char[] b) {
        return (b[0]==b[1] && b[1]==b[2]) ||
               (b[3]==b[4] && b[4]==b[5]) ||
               (b[6]==b[7] && b[7]==b[8]) ||
               (b[0]==b[3] && b[3]==b[6]) ||
               (b[1]==b[4] && b[4]==b[7]) ||
               (b[2]==b[5] && b[5]==b[8]) ||
               (b[0]==b[4] && b[4]==b[8]) ||
               (b[2]==b[4] && b[4]==b[6]);
    }

    // Check if all cells are filled
    static boolean isFull(char[] b) {
        for (char c : b) {
            if (c != 'X' && c != 'O') return false;
        }
        return true;
    }
}
