document.addEventListener('DOMContentLoaded', function() {
    var board, game = new Chess();

    var config = {
        draggable: true,
        position: 'start',
        onDrop: handleMove,
        onSnapEnd: updateBoard
    };

    board = Chessboard('myBoard', config);

    function handleMove(source, target) {
        var move = game.move({
            from: source,
            to: target,
            promotion: 'q'
        });

        if (move === null) return 'snapback';
        else window.setTimeout(makeBestMove, 250);
    }

    function updateBoard() {
        board.position(game.fen());
    }

    function makeBestMove() {
        var bestMove = getBestMove(game);
        game.move(bestMove);
        board.position(game.fen());
    }

    function getBestMove(game) {
        var moves = game.moves();
        var move = moves[Math.floor(Math.random() * moves.length)];
        return move;
    }
});
