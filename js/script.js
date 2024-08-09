// script.js
document.addEventListener('DOMContentLoaded', function () {
    var board,
        game = new Chess(),
        engine = STOCKFISH();

    // Set up the board
    board = Chessboard('board', {
        draggable: true,
        position: 'start',
        onDrop: handleMove
    });

    // Handle move
    function handleMove(source, target) {
        var move = game.move({
            from: source,
            to: target,
            promotion: 'q' // Always promote to a queen for simplicity
        });

        if (move === null) return 'snapback';

        window.setTimeout(makeAIMove, 250);
    }

    function makeAIMove() {
        engine.postMessage('position fen ' + game.fen());
        engine.postMessage('go movetime 1000');

        engine.onmessage = function (event) {
            var message = event.data;

            if (message.startsWith('bestmove')) {
                var move = message.split(' ')[1];
                game.move(move);
                board.position(game.fen());
            }
        };
    }
});

