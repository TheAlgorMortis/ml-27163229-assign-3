import math
import torch
from torch import nn
from rnns import ElmanRNN, JordanRNN, MultiRecurrentRNN

MODELS=[ElmanRNN,JordanRNN,MultiRecurrentRNN]
SEED=42

def test_forward(cls):
    torch.manual_seed(SEED)
    m=cls(input_size=1,hidden_size=16,output_size=1)
    x=torch.randn(8,24,1)
    y=m(x)
    ok=y.shape==(8,1)
    print(f"{cls.__name__:<20} forward={'PASS' if ok else 'FAIL'} output_shape={tuple(y.shape)}")
    return ok

def test_backward(cls):
    torch.manual_seed(SEED)
    m=cls(input_size=1,hidden_size=16,output_size=1)
    x=torch.randn(8,24,1)
    y=torch.randn(8,1)
    loss=nn.MSELoss()(m(x),y)
    loss.backward()
    ps=[p for p in m.parameters() if p.requires_grad]
    exists=all(p.grad is not None for p in ps)
    nonzero=any(p.grad is not None and torch.any(p.grad!=0) for p in ps)
    ok=exists and nonzero
    print(f"{cls.__name__:<20} backward={'PASS' if ok else 'FAIL'} loss={loss.item():.6f} gradients_exist={exists} gradients_nonzero={nonzero}")
    return ok

def test_weight_update(cls):
    torch.manual_seed(SEED)
    m=cls(input_size=1,hidden_size=16,output_size=1)
    opt=torch.optim.Adam(m.parameters(),lr=1e-3)
    x=torch.randn(8,24,1)
    y=torch.randn(8,1)
    before={n:p.detach().clone() for n,p in m.named_parameters()}
    opt.zero_grad()
    loss=nn.MSELoss()(m(x),y)
    loss.backward()
    opt.step()
    changed=any(not torch.equal(before[n],p.detach()) for n,p in m.named_parameters())
    print(f"{cls.__name__:<20} weight_update={'PASS' if changed else 'FAIL'} weights_changed={changed}")
    return changed

def test_reproducible_initialization(cls):
    torch.manual_seed(SEED)
    m1=cls(input_size=1,hidden_size=16,output_size=1)
    torch.manual_seed(SEED)
    m2=cls(input_size=1,hidden_size=16,output_size=1)
    same=all(torch.equal(p1,p2) for p1,p2 in zip(m1.parameters(),m2.parameters()))
    print(f"{cls.__name__:<20} reproducible_init={'PASS' if same else 'FAIL'} same_weights={same}")
    return same

def make_sine_data(n_points=300,window=20):
    t=torch.linspace(0,20*math.pi,n_points)
    s=torch.sin(t)
    X=torch.stack([s[i:i+window] for i in range(len(s)-window)]).unsqueeze(-1)
    y=torch.stack([s[i+window] for i in range(len(s)-window)]).unsqueeze(-1)
    return X,y

def test_overfit(cls):
    torch.manual_seed(SEED)
    X,y=make_sine_data()
    X,y=X[:64],y[:64]
    m=cls(input_size=1,hidden_size=16,output_size=1)
    opt=torch.optim.Adam(m.parameters(),lr=1e-2)
    loss_fn=nn.MSELoss()
    initial=None
    for epoch in range(500):
        opt.zero_grad()
        loss=loss_fn(m(X),y)
        loss.backward()
        opt.step()
        if epoch==0:
            initial=loss.item()
    final=loss.item()
    ok=final<initial*0.1
    print(f"{cls.__name__:<20} overfit={'PASS' if ok else 'FAIL'} initial_loss={initial:.6f} final_loss={final:.6f}")
    return ok

def run_all_tests():
    print("RNN smoke tests")
    print("="*80)
    all_ok=True
    for cls in MODELS:
        print(f"\n{cls.__name__}\n"+"-"*80)
        ok=all([test_forward(cls),test_backward(cls),test_weight_update(cls),test_reproducible_initialization(cls),test_overfit(cls)])
        all_ok=all_ok and ok
        print(f"{cls.__name__:<20} overall={'PASS' if ok else 'FAIL'}")
    print("\n"+"="*80)
    print("ALL SMOKE TESTS PASSED" if all_ok else "ONE OR MORE SMOKE TESTS FAILED")
    return all_ok

if __name__=="__main__":
    run_all_tests()
