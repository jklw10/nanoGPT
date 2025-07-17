import torch
import utils

q0 = torch.rand(10,10,10,device="cuda")
q1 = torch.rand(10,10,10,device="cuda")
q2 = torch.rand(10,10,10,device="cuda")
r1 = utils.fft_trunc_csquish(torch.cat((q0,q1,q2),dim=-1).to(torch.float32), 10)
r2 = utils.fft_trunc_csquish(torch.cat((q0,q1),dim=-1).to(torch.float32), 10)
r3 = utils.fft_trunc_csquish(torch.cat((r2,q2),dim=-1).to(torch.float32), 10)

r4 = utils.fft_trunc_csquish(torch.cat((q1,q0),dim=-1).to(torch.float32), 10)
r5 = utils.fft_trunc_csquish(torch.cat((q1,q2),dim=-1).to(torch.float32), 10)
r6 = utils.fft_trunc_csquish(torch.cat((q0,r5),dim=-1).to(torch.float32), 10)

print(f"collativity in result: {torch.nn.functional.mse_loss(r1,r3)}")
print(f"associativity in result: {torch.nn.functional.mse_loss(r2,r4)}")
print(f"commutativity in result: {torch.nn.functional.mse_loss(r6,r3)}")