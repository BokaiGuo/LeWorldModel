import pytest
from experiments.spectral_reachability import chat_p3_20261027_scheduler as s


@pytest.mark.parametrize('w', [[162.32405169996156,156.2867146609139],[1,2],[0,1],[1,0]])
def test_separation(w):
    d, r = s.allocation('D_rank',w), s.allocation('S_rank',w)
    assert d != r and d != [10,10] and r == d[::-1]
    for blocks in (d,r,[10,10]):
        jobs=s.queue(blocks)
        assert len(jobs)==60
        for slot in (0,1):
            assert [j['index'] for j in jobs if j['slot']==slot]==list(range(3*blocks[slot]))
        assert sum(j['slot']==0 for j in jobs[:30])==15


@pytest.mark.parametrize('w', [[0,0],[1,1]])
def test_tie_collision(w):
    assert s.allocation('D_rank',w)==s.allocation('S_rank',w)==[10,10]


@pytest.mark.parametrize('w', [[-1,2],[float('nan'),1],[float('inf'),1],[True,1],[]])
def test_bad(w):
    with pytest.raises(ValueError): s.allocation('D_rank',w)


def test_no_signal():
    with pytest.raises(ValueError): s.allocation('N',[1,2])
    assert s.allocation('N')==s.allocation('U')


def test_consumer_and_failure():
    jobs=s.queue([15,5])
    out=s.execute(jobs,lambda slot,index: float(index))
    assert out['complete'] and len(out['records'])==60
    failed=s.execute(jobs,lambda slot,index: 1/0)
    assert not failed['complete'] and len(failed['records'])==1 and len(failed['blocked'])==59
    assert s.consume_saved(jobs,[])['paired_gain'] is None
    with pytest.raises(ValueError):
        s.consume_saved(jobs,[{'slot':0,'index':0,'cost':1}]*2)
