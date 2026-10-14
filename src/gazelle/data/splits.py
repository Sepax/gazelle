from dataclasses import dataclass


@dataclass(frozen=True)
class Split:
    """Participant IDs for one leave-one-person-out fold."""

    train: list[int]
    test: list[int]


def lopo_split(test_subject: int, num_subjects: int = 15) -> Split:
    """Hold out one participant; train on the rest.

    Args:
        test_subject: Participant to hold out, in ``[0, num_subjects)``.
        num_subjects: Participant count; 15 for MPIIFaceGaze.

    Returns:
        Split with the remaining IDs in ``train``, ascending.

    Raises:
        ValueError: If ``test_subject`` is out of range.
    """
    if test_subject < 0 or test_subject >= num_subjects:
        raise ValueError(f"test_subject must be in [0, {num_subjects - 1}]")
    train_subjects = [i for i in range(num_subjects) if i != test_subject]
    return Split(train=train_subjects, test=[test_subject])
