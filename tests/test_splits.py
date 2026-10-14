import pytest

from gazelle.data.splits import lopo_split


def test_train_and_test_are_disjoint():
    split = lopo_split(7)
    assert set(split.train).isdisjoint(split.test)


def test_union_covers_every_participant():
    split = lopo_split(7)
    assert sorted(split.train + split.test) == list(range(15))


def test_holds_out_exactly_one_participant():
    split = lopo_split(7)
    assert split.test == [7]
    assert len(split.train) == 14


def test_train_is_sorted_ascending():
    assert lopo_split(7).train == sorted(lopo_split(7).train)


def test_small_num_subjects():
    assert lopo_split(0, num_subjects=4) == lopo_split(0, num_subjects=4)
    assert lopo_split(2, num_subjects=4).train == [0, 1, 3]


@pytest.mark.parametrize("test_subject", range(15))
def test_every_fold_is_valid(test_subject):
    split = lopo_split(test_subject)
    assert split.test == [test_subject]
    assert test_subject not in split.train


@pytest.mark.parametrize("bad", [-1, 15, 99])
def test_out_of_range_rejected(bad):
    with pytest.raises(ValueError):
        lopo_split(bad)
