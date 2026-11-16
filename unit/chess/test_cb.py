import torch
from models.chess_bot import GuardrailedTorchChess, uci_to_action

device = "cuda" if torch.cuda.is_available() else "cpu"
env = GuardrailedTorchChess(batch_size=4, device=device, halfmove_limit=100)
dones = torch.zeros(4, dtype=torch.bool, device=device)
winners = torch.zeros(4, dtype=torch.int8, device=device)

games_uci = [
    # 0: Draws on ply 8 (3rd occurrence of start pos)
    ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"],
    # 1: Irreversible e3/e6 resets history -> never hits 3 repetitions across 16 plies
    ["g1f3", "g8f6", "f3g1", "f6g8", "e2e3", "e7e6", "g1f3", "g8f6",
     "f3g1", "f6g8", "g1f3", "g8f6", "b1c3", "b8c6", "f1e2", "f8e7"],
    # 2: Loses K/k castling rights on ply 5 -> H(ply 8) != H(ply 4), draws only on ply 16
    ["e2e3", "e7e6", "g1f3", "g8f6", "h1g1", "h8g8", "g1h1", "g8h8",
     "f3g1", "f6g8", "g1f3", "f6g8", "f3g1", "f6g8", "g1f3", "g8f6"],
    # 3: Normal Ruy Lopez + castling -> never repeats
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6",
     "e1g1", "f8e7", "d2d3", "b7b5", "a4b3", "d7d6", "c2c3", "e8g8"],
]

# Fix typo in Game 2 ply 12 ('f6g8' -> 'g8f6')
games_uci[2][11] = "g8f6"

for ply in range(16):
    l_mask = env.compute_legal_masks()
    actions = torch.zeros(4, dtype=torch.long, device=device)

    for b in range(4):
        if not dones[b]:
            uci = games_uci[b][ply]
            act = uci_to_action(uci)
            assert l_mask[b, act] > -1e8, f"Game {b} ply {ply+1} illegal move {uci}"
            actions[b] = act

    dones, winners = env.step(actions, dones, winners, legal_mask=l_mask)
    print(f"Ply {ply+1:02d} | dones={dones.tolist()} | winners={winners.tolist()}")

    if ply + 1 == 4:
        assert not dones[0], "False positive: drew on 2nd repetition!"
    if ply + 1 == 8:
        assert dones[0] and winners[0] == 3, "Game 0 should draw on ply 8!"
        assert not dones[1], "Game 1 should NOT draw after e3/e6 reset!"
        assert not dones[2], "Game 2 should NOT draw (castling rights changed)!"
        assert not dones[3], "Game 3 should NOT draw!"
    if ply + 1 == 12:
        assert not dones[2], "Game 2 only reached post-rook-move pos twice by ply 12!"
    if ply + 1 == 16:
        assert not dones[1] and not dones[3], "Games 1 and 3 should still be active!"
        assert dones[2] and winners[2] == 3, "Game 2 should draw on ply 16 (3rd post-castling-loss pos)!"

print("\n[PASS] All Zobrist threefold repetition + castling/reset edge cases passed!")